import os
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from envs.maritime_medevac_env import MaritimeMedevacVecEnv
from agents.sd_drqn import SD_DRQN
from utils.episode_replay import EpisodeReplayBuffer, Step


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

def main(device: str = "cpu"):
    device = torch.device(device)

    # ----------------- Hyperparameters -----------------
    n_envs = 1
    n_agents = 2
    max_steps_per_episode = 20

    hidden_dim = 64
    gamma = 0.99
    lr = 5e-4  # slightly smaller LR for stability

    buffer_capacity_episodes = 1000
    batch_size = 32

    burn_in_default = 2
    learn_len_default = 8
    updates_per_episode_default = 4

    eps_start = 1.0
    eps_end = 0.10
    eps_decay_episodes = 4000  # slower decay; more exploration

    tau_max = 10
    seed = 0

    total_episodes = 5000
    warm_start_episodes = 0  # 0 => heuristic never used

    global_step = 0
    episode_returns = []

    run_name = f"sd_drqn_medevac_single_phase_{time.strftime('%Y%m%d_%H%M%S')}"
    log_dir = os.path.join("runs", run_name)
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = os.path.join("checkpoints", "sd_drqn_medevac_single_phase.pt")

    writer = SummaryWriter(log_dir=log_dir)

    # ----------------- Env & Agent -----------------
    env = MaritimeMedevacVecEnv(
        n_envs=n_envs,
        max_steps=max_steps_per_episode,
        tau_max=tau_max,
        n_agents=n_agents,
        device=device,
        seed=seed,
    )

    n_agents = env.n_agents
    n_obs = env.n_obs
    n_actions = env.n_actions
    state_dim = env.state_dim
    n_subsets = env.n_subsets

    agent = SD_DRQN(
        n_agents=n_agents,
        n_obs=n_obs,
        n_actions=n_actions,
        state_dim=state_dim,
        n_subsets=n_subsets,
        hidden_dim=hidden_dim,
        gamma=gamma,
        lr=lr,
        tau_max=tau_max,
        device=device,
    )

    buffer = EpisodeReplayBuffer(capacity_episodes=buffer_capacity_episodes)

    # For logging environment stats every 100 episodes
    prev_evac = 0
    prev_p_boat = 0
    prev_p_helo = 0

    # ----------------- Training loop -----------------
    for ep in range(1, total_episodes + 1):
        data = env.reset()
        obs = data["obs"]
        state = data["state"]
        tau_vec = data["tau_vec"]
        comm_mask = data["comm_mask"]
        done = data["done"]

        agent.init_exec_hidden(n_envs)

        episode_steps = []
        ep_ret = 0.0

        # Epsilon schedule
        eps = np.interp(
            ep,
            [1, eps_decay_episodes],
            [eps_start, eps_end],
        )

        # Phase-based training schedule
        if ep < 1500:
            burn_in = 1
            learn_len = 6
            updates_per_episode = 2
        elif ep < 3500:
            burn_in = 2
            learn_len = 8
            updates_per_episode = 4
        else:
            burn_in = 2
            learn_len = 10
            updates_per_episode = 6

        # Log the active schedule to TensorBoard
        writer.add_scalar("train/burn_in", burn_in, ep)
        writer.add_scalar("train/learn_len", learn_len, ep)
        writer.add_scalar("train/updates_per_episode", updates_per_episode, ep)

        use_heuristic = ep <= warm_start_episodes  # currently always False

        while not done.all():
            if use_heuristic:
                actions = heuristic_medevac_actions(env)
            else:
                actions = agent.select_actions(obs, eps=eps)

            next_data, reward = env.step(actions)

            next_obs = next_data["obs"]
            next_state = next_data["state"]
            next_tau_vec = next_data["tau_vec"]
            next_comm_mask = next_data["comm_mask"]
            done = next_data["done"]

            r_val = float(reward[0].item())
            ep_ret += r_val

            e = 0  # single env
            step = Step(
                obs=obs[e].cpu().tolist(),
                state=state[e].cpu().tolist(),
                tau_vec=tau_vec[e].cpu().tolist(),
                comm_mask=comm_mask[e].cpu().tolist(),
                actions=actions[e].cpu().tolist(),
                reward=r_val,
                next_obs=next_obs[e].cpu().tolist(),
                next_state=next_state[e].cpu().tolist(),
                next_tau_vec=next_tau_vec[e].cpu().tolist(),
                next_comm_mask=next_comm_mask[e].cpu().tolist(),
                done=float(done[e].item()),
            )
            episode_steps.append(step)

            obs = next_obs
            state = next_state
            tau_vec = next_tau_vec
            comm_mask = next_comm_mask

            global_step += 1

        buffer.push_episode(episode_steps)
        episode_returns.append(ep_ret)
        writer.add_scalar("reward/episode_return", ep_ret, ep)

        # Replay training
        if buffer.num_episodes() > 5:
            for _ in range(updates_per_episode):
                loss_total, loss_dec, loss_cen = agent.train_step(
                    buffer,
                    batch_size=batch_size,
                    burn_in=burn_in,
                    learn_len=learn_len,
                )
                if loss_total is not None:
                    writer.add_scalar("loss/total", loss_total, ep)
                    writer.add_scalar("loss/dec", loss_dec, ep)
                    writer.add_scalar("loss/cen", loss_cen, ep)

        avg_last_50 = (
            np.mean(episode_returns[-50:])
            if len(episode_returns) >= 50
            else np.mean(episode_returns)
        )
        src = "H" if use_heuristic else "L"
        print(
            f"Ep {ep:4d} | ret: {ep_ret:6.2f} | avg_last_50: {avg_last_50:6.2f} "
            f"| eps: {eps:5.3f} | src={src}"
        )

        if ep % 200 == 0:
            agent.save_checkpoint(ckpt_path, global_step)
            print(f"Saved interim checkpoint to {ckpt_path}")

        # Environment stats every 100 episodes
        if ep % 100 == 0 and ep > 0:
            block_evac = env.stats_evacuations - prev_evac
            block_p_boat = env.stats_pickups_boat - prev_p_boat
            block_p_helo = env.stats_pickups_helo - prev_p_helo

            rate_evac = block_evac / 100.0
            rate_p_boat = block_p_boat / 100.0
            rate_p_helo = block_p_helo / 100.0

            print(
                f"[ENV STATS] ep={ep} "
                f"block_evacs={block_evac} block_boat={block_p_boat} block_helo={block_p_helo} | "
                f"total_evacs={env.stats_evacuations} "
                f"total_boat={env.stats_pickups_boat} total_helo={env.stats_pickups_helo}"
            )

            writer.add_scalar("env/evacs_per_100eps", rate_evac, ep)
            writer.add_scalar("env/boat_pickups_per_100eps", rate_p_boat, ep)
            writer.add_scalar("env/helo_pickups_per_100eps", rate_p_helo, ep)

            prev_evac = env.stats_evacuations
            prev_p_boat = env.stats_pickups_boat
            prev_p_helo = env.stats_pickups_helo

    writer.close()
    agent.save_checkpoint(ckpt_path, global_step)
    print(f"Training complete. Final checkpoint saved to {ckpt_path}")

# ---------------------------------------------------------------------
# Heuristic controller (if used, otherwise – warm_start_episodes = 0)
# ---------------------------------------------------------------------


def _move_towards(env: MaritimeMedevacVecEnv, curr_pos: int, target_pos: int) -> int:
    device = env.device
    px, py = env._idx_to_xy(torch.tensor(curr_pos, device=device))
    tx, ty = env._idx_to_xy(torch.tensor(target_pos, device=device))

    if px < tx:
        return MaritimeMedevacVecEnv.ACT_E
    if px > tx:
        return MaritimeMedevacVecEnv.ACT_W
    if py < ty:
        return MaritimeMedevacVecEnv.ACT_S
    if py > ty:
        return MaritimeMedevacVecEnv.ACT_N
    return MaritimeMedevacVecEnv.ACT_HOLD


def heuristic_medevac_actions(env: MaritimeMedevacVecEnv) -> torch.LongTensor:
    device = env.device
    E = env.n_envs
    acts = torch.full(
        (E, env.n_agents),
        MaritimeMedevacVecEnv.ACT_HOLD,
        dtype=torch.long,
        device=device,
    )

    for e in range(E):
        helo_pos = int(env.helo_pos[e].item())
        boat_pos = int(env.boat_pos[e].item())
        helo_has = int(env.helo_has_patient[e].item())
        boat_has = int(env.boat_has_patient[e].item())
        pier_count = int(env.pier_count[e].item())
        island_counts = [
            int(env.island_counts[e, k].item()) for k in range(env.n_islands)
        ]

        # ------------- Boat policy (agent 1) -------------
        if boat_has == 0:
            # Load if on island with casualties
            if boat_pos in env.island_pos:
                idx = env.island_pos.index(boat_pos)
                if island_counts[idx] > 0:
                    acts[e, 1] = MaritimeMedevacVecEnv.ACT_LOAD
                    continue

            # Otherwise go to nearest island with casualties
            target_island = None
            best_dist = 999
            for k, pos in enumerate(env.island_pos):
                if island_counts[k] > 0:
                    px, py = env._idx_to_xy(torch.tensor(boat_pos, device=device))
                    ix, iy = env._idx_to_xy(torch.tensor(pos, device=device))
                    d = int((px - ix).abs().item() + (py - iy).abs().item())
                    if d < best_dist:
                        best_dist = d
                        target_island = pos

            if target_island is not None:
                acts[e, 1] = _move_towards(env, boat_pos, target_island)
            else:
                # No more island patients: loiter at pier
                if boat_pos != env.pier_pos:
                    acts[e, 1] = _move_towards(env, boat_pos, env.pier_pos)
                else:
                    acts[e, 1] = MaritimeMedevacVecEnv.ACT_HOLD
        else:
            # Boat has patient: go to pier and unload
            if boat_pos == env.pier_pos:
                acts[e, 1] = MaritimeMedevacVecEnv.ACT_UNLOAD
            else:
                acts[e, 1] = _move_towards(env, boat_pos, env.pier_pos)

        # ------------- Helo policy (agent 0) -------------
        if helo_has == 0:
            # Load at pier if patients are there
            if helo_pos == env.pier_pos and pier_count > 0:
                acts[e, 0] = MaritimeMedevacVecEnv.ACT_LOAD
                continue

            # Load at island if casualties present
            if helo_pos in env.island_pos:
                idx = env.island_pos.index(helo_pos)
                if island_counts[idx] > 0:
                    acts[e, 0] = MaritimeMedevacVecEnv.ACT_LOAD
                    continue

            # Otherwise, move to pier if it has patients, else nearest island, else hospital
            if pier_count > 0:
                target = env.pier_pos
            else:
                target = None
                best_dist = 999
                for k, pos in enumerate(env.island_pos):
                    if island_counts[k] > 0:
                        px, py = env._idx_to_xy(torch.tensor(helo_pos, device=device))
                        ix, iy = env._idx_to_xy(torch.tensor(pos, device=device))
                        d = int((px - ix).abs().item() + (py - iy).abs().item())
                        if d < best_dist:
                            best_dist = d
                            target = pos
                if target is None:
                    target = env.hospital_pos

            acts[e, 0] = _move_towards(env, helo_pos, target)
        else:
            # Helo has patient: fly to hospital and unload
            if helo_pos == env.hospital_pos:
                acts[e, 0] = MaritimeMedevacVecEnv.ACT_UNLOAD
            else:
                acts[e, 0] = _move_towards(env, helo_pos, env.hospital_pos)

    return acts

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    main(device=device)