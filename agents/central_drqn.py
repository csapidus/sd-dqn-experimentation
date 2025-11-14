import os
import torch
import torch.nn as nn
from torch.optim import Adam

from .agent_drqnet import AgentDRQNet


class CentralDRQN(nn.Module):
    """
    Pure centralized DRQN for Dec-Tiger.

    - Single agent that sees a JOINT observation index
      (o0, o1) ∈ {0,1,2}^2  ->  obs_joint ∈ {0..8}
    - Takes a JOINT action index
      (a0, a1) ∈ {0,1,2}^2  ->  act_joint ∈ {0..8}
    """

    def __init__(
        self,
        n_joint_obs: int,
        n_joint_actions: int,
        gamma: float = 0.99,
        lr: float = 5e-4,
        hidden_dim: int = 64,
        device: str = "cpu",
    ):
        super().__init__()
        self.n_joint_obs = n_joint_obs
        self.n_joint_actions = n_joint_actions
        self.gamma = gamma
        self.device = device
        self.hidden_dim = hidden_dim

        # Online + target DRQN
        self.online = AgentDRQNet(n_joint_obs, n_joint_actions, hidden_dim).to(device)
        self.target = AgentDRQNet(n_joint_obs, n_joint_actions, hidden_dim).to(device)
        self.target.load_state_dict(self.online.state_dict())

        self.optimizer = Adam(self.online.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        # Hidden state for execution
        self.exec_hidden = None  # [1, E, H]

    # ------------- execution hidden state -------------

    def init_exec_hidden(self, n_envs: int):
        self.exec_hidden = torch.zeros(1, n_envs, self.hidden_dim, device=self.device)

    def reset_exec_hidden(self, env_indices=None):
        if self.exec_hidden is None:
            return
        if env_indices is None:
            self.exec_hidden.zero_()
        else:
            self.exec_hidden[:, env_indices, :].zero_()

    # ------------- epsilon-greedy action selection -------------

    def select_actions(self, obs_joint, eps: float):
        """
        obs_joint: LongTensor [E] with values in [0, n_joint_obs-1]
        returns: LongTensor [E] joint actions in [0, n_joint_actions-1]
        """
        E = obs_joint.shape[0]
        if self.exec_hidden is None or self.exec_hidden.shape[1] != E:
            self.init_exec_hidden(E)

        rand = torch.rand(E, device=self.device)
        explore = rand < eps

        q_t, h_new = self.online.forward_step(obs_joint, self.exec_hidden)
        self.exec_hidden = h_new
        greedy_actions = q_t.argmax(dim=-1)

        random_actions = torch.randint(
            low=0, high=self.n_joint_actions, size=(E,), device=self.device
        )

        actions = torch.where(explore, random_actions, greedy_actions)
        return actions

    # ------------- soft target update -------------

    @torch.no_grad()
    def update_target(self, tau: float = 0.01):
        for tgt_p, p in zip(self.target.parameters(), self.online.parameters()):
            tgt_p.data.copy_(tau * p.data + (1.0 - tau) * tgt_p.data)

    # ------------- training -------------

    def train_step(self, replay_buffer, batch_size: int, burn_in: int, learn_len: int):
        """
        DRQN training with burn-in + learn window (R2D2-style).

        - Sample sequences of length T = burn_in + learn_len.
        - Unroll ONLINE net on obs[0:T].
        - Unroll TARGET net on next_obs[0:T].
        - 1-step TD loss on last `learn_len` timesteps after burn-in.
        """
        if replay_buffer.num_episodes() == 0:
            return None

        device = self.device
        T = burn_in + learn_len

        batch = replay_buffer.sample_sequences(batch_size, T)

        def to_tensor(x, dtype):
            return torch.tensor(x, dtype=dtype, device=device)

        obs = to_tensor(batch["obs"], torch.long)          # [B, T]
        next_obs = to_tensor(batch["next_obs"], torch.long)
        actions = to_tensor(batch["actions"], torch.long)  # [B, T]
        reward = to_tensor(batch["reward"], torch.float32) # [B, T]
        done = to_tensor(batch["done"], torch.float32)     # [B, T]

        B, T_actual = obs.shape
        if T_actual < T:
            # Not enough length yet to support full window
            return None

        learn_start = burn_in
        learn_end = burn_in + learn_len  # exclusive
        K = learn_len

        # ---- ONLINE unroll ----
        q_seq, _ = self.online.forward_seq(obs)  # [B, T, A]
        a_seq = actions.unsqueeze(-1)            # [B, T, 1]
        q_taken = q_seq.gather(2, a_seq).squeeze(-1)  # [B, T]

        q_learn = q_taken[:, learn_start:learn_end]        # [B, K]
        r_learn = reward[:, learn_start:learn_end]         # [B, K]
        done_learn = done[:, learn_start:learn_end]        # [B, K]

        # ---- TARGET unroll ----
        with torch.no_grad():
            q_next_seq, _ = self.target.forward_seq(next_obs)  # [B, T, A]
            next_actions = q_next_seq.argmax(dim=-1, keepdim=True)  # [B, T, 1]
            q_next_taken = q_next_seq.gather(2, next_actions).squeeze(-1)  # [B, T]
            q_next_learn = q_next_taken[:, learn_start:learn_end]          # [B, K]

            target = r_learn + self.gamma * (1.0 - done_learn) * q_next_learn

        loss = self.loss_fn(q_learn, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()

        return float(loss.item())

    # ------------- checkpointing -------------

    def save_checkpoint(self, path: str, step: int):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            "step": step,
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(state, path)

    def load_checkpoint(self, path: str):
        if not os.path.isfile(path):
            print(f"[CentralDRQN] No checkpoint found at {path}")
            return 0
        state = torch.load(path, map_location=self.device)
        self.online.load_state_dict(state["online"])
        self.target.load_state_dict(state["target"])
        self.optimizer.load_state_dict(state["optimizer"])
        print(f"[CentralDRQN] Loaded checkpoint from {path}, step={state['step']}")
        return state["step"]
