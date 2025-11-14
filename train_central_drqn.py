import os
import time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from envs.tiger_comm_vec import TigerCommVecEnv
from agents.central_drqn import CentralDRQN
from utils.central_episode_replay import CentralEpisodeReplayBuffer, CentralStep


# ---------------------------------------------------------------
# Encode joint observation (2 agents -> 3^2 = 9 states)
# ---------------------------------------------------------------
def encode_joint_obs(obs_agents, n_obs_per_agent):
    E, n_agents = obs_agents.shape
    powers = torch.tensor(
        [n_obs_per_agent ** i for i in range(n_agents - 1, -1, -1)],
        device=obs_agents.device,
        dtype=torch.long,
    )
    return (obs_agents * powers).sum(dim=-1)


# ---------------------------------------------------------------
# Decode joint action index -> per-agent actions
# ---------------------------------------------------------------
def decode_joint_action(a_joint, n_actions_per_agent, n_agents):
    E = a_joint.shape[0]
    acts = torch.zeros(E, n_agents, dtype=torch.long, device=a_joint.device)
    x = a_joint.clone()
    for i in range(n_agents - 1, -1, -1):
        acts[:, i] = x % n_actions_per_agent
        x = x // n_actions_per_agent
    return acts


# ---------------------------------------------------------------
# Structured teacher exploration: listen-listen-open
# ---------------------------------------------------------------
def teacher_open(obs_joint, n_actions_per_agent, n_agents, device):
    E = obs_joint.shape[0]
    # decode to per-agent obs
    obs_agents = torch.zeros(E, n_agents, dtype=torch.long, device=device)
    x = obs_joint.clone()
    for i in range(n_agents - 1, -1, -1):
        obs_agents[:, i] = x % 3
        x = x // 3

    # majority vote
    mean_obs = obs_agents.float().mean(dim=1)
    # 0=left, 1=right, 2=no_obs
    a_joint = torch.zeros(E, dtype=torch.long, device=device)

    for e in range(E):
        if mean_obs[e] == 0:      # heard left -> tiger left -> open right = 2
            aa = [2, 2]
        elif mean_obs[e] == 1:    # heard right -> open left = 1
            aa = [1, 1]
        else:
            aa = torch.randint(0, 3, (2,), device=device)
        a_joint[e] = aa[0] * 3 + aa[1]

    return a_joint


# ---------------------------------------------------------------
# TRAINING SCRIPT
# ---------------------------------------------------------------
def main(device="cpu"):
    device = torch.device(device)

    # ------------------------
    # Hyperparameters
    # ------------------------
    total_episodes = 5000

    n_envs = 1
    listen_correct_prob = 0.85
    max_steps = 20
    tau_max = 10
    n_agents = 2

    # DRQN HPs
    gamma = 0.99
    lr = 1e-4
    hidden_dim = 64
    burn_in = 4
    learn_len = 4
    batch_size = 32
    updates_per_episode = 16
    buffer_capacity = 2000

    # ε schedule (keep high for long time)
    eps_start = 1.0
    eps_end = 0.1
    eps_decay = 5000

    # α schedule for reward homotopy
    # α=1 → shaped reward only
    # α=0 → true reward only
    def alpha(ep):
        if ep < 2000:
            return 1.0
        if ep >= 4000:
            return 0.0
        return 1.0 - ((ep - 2000) / 2000.0)

    # teacher forcing probability
    teacher_prob = 0.15  # 15% of episodes use structured exploration

    # ------------------------
    # Build env/agent/buffer
    # ------------------------
    env = TigerCommVecEnv(
        n_envs=n_envs,
        listen_correct_prob=listen_correct_prob,
        max_steps=max_steps,
        tau_max=tau_max,
        n_agents=n_agents,
        device=device,
        seed=42,
    )

    n_obs_per_agent = env.n_obs
    n_joint_obs = n_obs_per_agent ** n_agents
    n_actions_per_agent = env.n_actions
    n_joint_actions = n_actions_per_agent ** n_agents

    agent = CentralDRQN(
        n_joint_obs=n_joint_obs,
        n_joint_actions=n_joint_actions,
        gamma=gamma,
        lr=lr,
        hidden_dim=hidden_dim,
        device=device,
    )

    buffer = CentralEpisodeReplayBuffer(buffer_capacity)

    # Logging
    run = f"central_drqn_alpha_mix_{time.strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=f"runs/{run}")
    ckpt = f"checkpoints/{run}.pt"

    total_gold = 0
    total_tiger = 0
    global_step = 0
    episode_returns_true = []

    # ============================================================
    # MAIN TRAINING LOOP
    # ============================================================
    for ep in range(1, total_episodes + 1):

        data = env.reset()
        agent.reset_exec_hidden()

        obs_agents = data["obs"]
        done = data["done"]
        obs_joint = encode_joint_obs(obs_agents, n_obs_per_agent)

        # Episode variables
        r_true_sum = 0.0
        last_r_true = 0.0
        episode_steps = []

        # compute exploration rate
        eps = max(eps_end, eps_start - (eps_start - eps_end) * (ep / eps_decay))

        structured = (np.random.rand() < teacher_prob)

        t = 0
        while not done.all():

            # ----------------------------
            # Select action
            # ----------------------------
            if structured:
                # teacher: listen twice then open
                if t < 2:
                    # both agents listen -> joint action index 0
                    a_joint = torch.zeros(n_envs, dtype=torch.long, device=device)
                else:
                    a_joint = teacher_open(
                        obs_joint, n_actions_per_agent, n_agents, device
                    )
            else:
                # normal exploration
                a_joint = agent.select_actions(obs_joint, eps)

            acts_agents = decode_joint_action(a_joint,
                                              n_actions_per_agent,
                                              n_agents)

            # ----------------------------
            # Environment step
            # ----------------------------
            next_data, r_true, r_shaped = env.step(acts_agents)

            # Mixture reward for training
            A = alpha(ep)
            r_train = (A * r_shaped + (1 - A) * r_true)

            next_obs_agents = next_data["obs"]
            done = next_data["done"]
            next_obs_joint = encode_joint_obs(next_obs_agents, n_obs_per_agent)

            r_true_sum += float(r_true[0].item())
            last_r_true = float(r_true[0].item())

            # store step
            step = CentralStep(
                obs=int(obs_joint[0].item()),
                actions=int(a_joint[0].item()),
                reward=float(r_train[0].item()),
                next_obs=int(next_obs_joint[0].item()),
                done=float(done[0].item()),
            )
            episode_steps.append(step)

            obs_joint = next_obs_joint
            t += 1

        # ----------------------------
        # Store episode + stats
        # ----------------------------
        buffer.push_episode(episode_steps)
        episode_returns_true.append(r_true_sum)
        writer.add_scalar("reward/true_episode_return", r_true_sum, ep)

        # open outcome
        if last_r_true > 0:
            total_gold += 1
        elif last_r_true < 0:
            total_tiger += 1

        # ----------------------------
        # Learn from replay
        # ----------------------------
        if buffer.num_episodes() > 5:
            for _ in range(updates_per_episode):
                loss = agent.train_step(
                    buffer,
                    batch_size,
                    burn_in,
                    learn_len,
                )
                if loss is not None:
                    writer.add_scalar("loss/td_loss", loss, global_step)
                    global_step += 1

        agent.update_target(tau=0.01)

        # ----------------------------
        # Logging
        # ----------------------------
        if ep % 10 == 0:
            last50 = episode_returns_true[-50:] if len(episode_returns_true) >= 50 else episode_returns_true
            success = total_gold / max(1, (total_gold + total_tiger))
            print(
                f"Ep {ep:4d} | ret_true: {r_true_sum:6.2f} | "
                f"avg_last50: {np.mean(last50):6.2f} | "
                f"eps: {eps:.3f} | "
                f"A={A:.2f} | "
                f"success(gold|door): {success:.3f} "
                f"(gold={total_gold}, tiger={total_tiger})"
            )

        if ep % 200 == 0:
            agent.save_checkpoint(ckpt, global_step)
            print(f"Saved checkpoint {ckpt}")

    writer.close()
    agent.save_checkpoint(ckpt, global_step)
    print(f"Training complete. Saved final checkpoint to {ckpt}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    main(device=device)
