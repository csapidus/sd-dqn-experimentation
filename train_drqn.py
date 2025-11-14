import os
import time
from copy import deepcopy

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from envs.tiger_comm_vec import TigerCommVecEnv
from agents.sd_drqn import SD_DRQN
from utils.episode_replay import EpisodeReplayBuffer, Step


@torch.no_grad()
def evaluate_policy(
    agent,
    base_env_kwargs: dict,
    device,
    n_eval_episodes: int,
    force_listen_eval: int,
    writer: SummaryWriter,
    global_step: int,
):

    from envs.tiger_comm_vec import TigerCommVecEnv

    eval_kwargs = deepcopy(base_env_kwargs)
    eval_kwargs["eta"] = 0.0          # canonical reward only during eval
    eval_kwargs["n_envs"] = 1         # eval with a single env for clarity

    env_eval = TigerCommVecEnv(**eval_kwargs)

    agent.eval()

    returns = []
    gold_count = 0
    tiger_count = 0

    T_eval = 0.05  # low temp => almost greedy softmax over Q

    for ep in range(n_eval_episodes):
        data = env_eval.reset()
        obs = data["obs"]
        done = data["done"]

        agent.init_exec_hidden(env_eval.n_envs)

        ep_ret = 0.0
        t = 0

        while not done.all():
            # Greedy-ish policy: eps=0, softmax_T=T_eval
            actions = agent.select_actions(obs, eps=0.0, softmax_T=T_eval)

            # Optionally force initial listening during eval
            if t < force_listen_eval:
                actions[:] = 0  # 0 = LISTEN

            next_data, reward = env_eval.step(actions)

            ep_ret += reward.mean().item()
            obs = next_data["obs"]
            done = next_data["done"]
            t += 1

        returns.append(ep_ret)
        if ep_ret > 0:
            gold_count += 1
        elif ep_ret < 0:
            tiger_count += 1

    agent.train()

    mean_ret = float(np.mean(returns))
    total_opens = gold_count + tiger_count
    success = gold_count / total_opens if total_opens > 0 else 0.0

    writer.add_scalar("eval/return", mean_ret, global_step)
    writer.add_scalar("eval/success_gold", success, global_step)

    print(
        f"[EVAL] step={global_step} | "
        f"mean_ret={mean_ret:.2f} | "
        f"success_gold={success:.3f} "
        f"(gold={gold_count}, tiger={tiger_count})"
    )


def main(device: str = "cpu"):
    # ----------------- Global hyperparameters -----------------
    device = torch.device(device)

    # Env hyperparameters
    n_envs = 1
    listen_correct_prob = 0.85
    max_steps_per_episode = 20
    tau_max = 10
    n_agents = 2
    seed = 42
    eta = 3.0  # intrinsic belief/entropy bonus scale for TRAINING

    # DRQN / optimization hyperparameters
    gamma = 0.99
    lr = 5e-5
    hidden_dim = 64

    buffer_capacity = 1000  # episodes
    batch_size = 32

    burn_in = 0
    learn_len = 8
    updates_per_episode = 16  # used when force_listen > 1

    total_episodes = 2000

    # Curriculum: forced LISTEN from 8 -> 1 over first ~800 episodes
    initial_force_listen = 8
    min_force_listen = 1
    curriculum_episodes = 800

    # Epsilon schedule
    eps_start = 1.0
    eps_end = 0.05
    eps_decay_episodes = 800

    # Softmax (Boltzmann) exploration schedule
    T_start = 1.5
    T_end = 0.05
    T_decay_episodes = 800

    # Logging / checkpoints
    run_name = f"sd_drqn_tiger_comm_single_phase_{time.strftime('%Y%m%d_%H%M%S')}"
    log_dir = os.path.join("runs", run_name)
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = os.path.join("checkpoints", "sd_drqn_tiger_comm_single_phase.pt")

    writer = SummaryWriter(log_dir=log_dir)

    # ----------------- Build environment & agent -----------------
    env = TigerCommVecEnv(
        n_envs=n_envs,
        listen_correct_prob=listen_correct_prob,
        max_steps=max_steps_per_episode,
        tau_max=tau_max,
        n_agents=n_agents,
        device=device,
        seed=seed,
        eta=eta,
    )

    base_env_kwargs = dict(
        n_envs=n_envs,
        listen_correct_prob=listen_correct_prob,
        max_steps=max_steps_per_episode,
        tau_max=tau_max,
        n_agents=n_agents,
        device=device,
        seed=seed,
        eta=eta,
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
        tau_max=tau_max,
        gamma=gamma,
        lr=lr,
        device=device,
        hidden_dim=hidden_dim,
        # central_loss_coef can be tuned; 0.1 is a reasonable starting point
        central_loss_coef=0.1,
    ).to(device)

    buffer = EpisodeReplayBuffer(buffer_capacity)

    # ----------------- Training loop -----------------
    global_step = 0
    episode_rewards = []
    num_tiger = 0
    num_gold = 0

    eval_interval = 200
    n_eval_episodes = 200
    force_listen_eval = 1  # or 0 if you want fully unconstrained eval

    for ep in range(1, total_episodes + 1):
        # Curriculum: forced LISTEN 8 -> 1, then stay at 1
        frac = min(ep / float(curriculum_episodes), 1.0)
        decayed = int(round(initial_force_listen * (1.0 - frac)))
        current_force_listen = max(min_force_listen, decayed)

        # Gentler optimization once we’re at force_listen = 1
        if current_force_listen == 1:
            updates_this_episode = max(updates_per_episode // 2, 4)
        else:
            updates_this_episode = updates_per_episode

        data = env.reset()
        agent.reset_exec_hidden()

        obs = data["obs"]          # [E, n_agents]
        state = data["state"]
        tau_vec = data["tau_vec"]
        comm_mask = data["comm_mask"]
        done = data["done"]

        ep_reward = 0.0
        episode_steps = []
        last_r_val = 0.0

        # epsilon & softmax temperature for this episode
        eps = max(
            eps_end,
            eps_start - (eps_start - eps_end) * (ep / float(eps_decay_episodes)),
        )
        T = max(
            T_end,
            T_start - (T_start - T_end) * (ep / float(T_decay_episodes)),
        )

        t = 0

        while not done.all():
            actions = agent.select_actions(obs, eps=eps, softmax_T=T)

            # Curriculum: first `current_force_listen` steps, force LISTEN
            if t < current_force_listen:
                actions = torch.zeros_like(actions)  # 0 = LISTEN

            next_data, reward = env.step(actions)

            next_obs = next_data["obs"]
            next_state = next_data["state"]
            next_tau_vec = next_data["tau_vec"]
            next_comm_mask = next_data["comm_mask"]
            done = next_data["done"]

            r_val = float(reward[0].item())
            ep_reward += r_val
            last_r_val = r_val

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
            t += 1

        # Store full episode
        buffer.push_episode(episode_steps)
        episode_rewards.append(ep_reward)
        writer.add_scalar("reward/episode_return", ep_reward, ep)

        # classify episode outcome using final reward
        if last_r_val > 0:
            num_gold += 1
        elif last_r_val < 0:
            num_tiger += 1

        # DRQN training from replay
        if buffer.num_episodes() > 5:
            for _ in range(updates_this_episode):
                result = agent.train_step(
                    buffer,
                    batch_size=batch_size,
                    burn_in=burn_in,
                    learn_len=learn_len,
                )
                if result is None:
                    continue

                loss_total, loss_dec, loss_cen = result
                writer.add_scalar("loss/total", loss_total, global_step)
                writer.add_scalar("loss/dec",   loss_dec,   global_step)
                writer.add_scalar("loss/cen",   loss_cen,   global_step)
                global_step += 1

        # Periodic target network updates
        if ep % 50 == 0:
            agent.update_targets()

        # Logging
        if ep % 10 == 0:
            last_50 = episode_rewards[-50:] if len(episode_rewards) >= 50 else episode_rewards
            avg_last_50 = np.mean(last_50)
            total_ends = num_gold + num_tiger
            success_rate = num_gold / total_ends if total_ends > 0 else 0.0

            print(
                f"Ep {ep:4d} | ret: {ep_reward:7.2f} | "
                f"avg_last_50: {avg_last_50:7.2f} | eps: {eps:.3f} | "
                f"buf_eps: {buffer.num_episodes():4d} | "
                f"success(gold|door): {success_rate:.3f} "
                f"(gold={num_gold}, tiger={num_tiger}, "
                f"force_listen={current_force_listen})"
            )

        # Periodic checkpoint
        if ep % 200 == 0:
            agent.save_checkpoint(ckpt_path, global_step)
            print(f"Saved interim checkpoint to {ckpt_path}")

        # Periodic deterministic eval
        if ep % eval_interval == 0:
            evaluate_policy(
                agent,
                base_env_kwargs,
                device,
                n_eval_episodes,
                force_listen_eval,
                writer,
                global_step,
            )

    writer.close()
    agent.save_checkpoint(ckpt_path, global_step)
    print(f"Training complete. Final checkpoint saved to {ckpt_path}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    main(device=device)
