import os
import torch
import torch.nn as nn
from torch.optim import Adam

from .agent_drqnet import AgentDRQNet
from .mixer_tau_subset import TauSubsetGatedMixer


class SD_DRQN(nn.Module):
    """Semi-Decentralized DRQN with τ-gated subset-aware mixer.

    - Per-agent DRQNs produce Q_i.
    - TauSubsetGatedMixer mixes per-agent Q_i into a joint Q_tot using τ and
      the comm subset.
    - Training uses R2D2-style burn-in + learn window.
    """

    def __init__(
        self,
        n_agents: int,
        n_obs: int,
        n_actions: int,
        state_dim: int,
        n_subsets: int,
        tau_max: int,
        gamma: float = 0.99,
        lr: float = 5e-4,
        device: str = "cpu",
        hidden_dim: int = 64,
        central_aux_weight: float = 0.1,
        central_loss_coef: float = 0.1,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.n_obs = n_obs
        self.n_actions = n_actions
        self.state_dim = state_dim
        self.n_subsets = n_subsets
        self.tau_max = tau_max
        self.gamma = gamma
        self.device = device
        self.hidden_dim = hidden_dim
        self.central_aux_weight = central_aux_weight
        self.central_loss_coef = central_loss_coef

        # Per-agent DRQNs
        self.agents = nn.ModuleList(
            [AgentDRQNet(n_obs, n_actions, hidden_dim).to(device) for _ in range(n_agents)]
        )
        self.target_agents = nn.ModuleList(
            [AgentDRQNet(n_obs, n_actions, hidden_dim).to(device) for _ in range(n_agents)]
        )
        for i in range(n_agents):
            self.target_agents[i].load_state_dict(self.agents[i].state_dict())

        # τ-gated mixer + target mixer
        self.mixer = TauSubsetGatedMixer(n_agents, state_dim, n_subsets).to(device)
        self.target_mixer = TauSubsetGatedMixer(n_agents, state_dim, n_subsets).to(device)
        self.target_mixer.load_state_dict(self.mixer.state_dict())

        # central joint head for future ablations
        joint_action_dim = n_actions ** n_agents
        central_in_dim = hidden_dim * n_agents + state_dim + n_subsets + n_agents
        self.central_head = nn.Sequential(
            nn.Linear(central_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, joint_action_dim),
        ).to(device)
        self.target_central_head = nn.Sequential(
            nn.Linear(central_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, joint_action_dim),
        ).to(device)
        self.target_central_head.load_state_dict(self.central_head.state_dict())

        # Single optimizer: agents + mixer + (optionally) central head
        self.params = (
            list(self.agents.parameters())
            + list(self.mixer.parameters())
            + list(self.central_head.parameters())
        )
        self.optimizer = Adam(self.params, lr=lr)
        self.loss_fn = nn.MSELoss()

        # Hidden states for execution (per-agent, per-env)
        self.exec_hidden = None  # [n_agents][1, E, H]

    # ------------------------------------------------------------------
    # Execution helpers
    # ------------------------------------------------------------------
    def init_exec_hidden(self, n_envs: int):
        self.exec_hidden = [
            torch.zeros(1, n_envs, self.hidden_dim, device=self.device)
            for _ in range(self.n_agents)
        ]

    def reset_exec_hidden(self, env_indices=None):
        if self.exec_hidden is None:
            return
        if env_indices is None:
            for h in self.exec_hidden:
                h.zero_()
        else:
            for h in self.exec_hidden:
                h[:, env_indices, :].zero_()

    def select_actions(self, obs, eps: float, softmax_T: float | None = None):
        """
        Epsilon-greedy per-agent action selection with recurrent state.

        obs: LongTensor [E, n_agents]
        returns: LongTensor [E, n_agents]

        If softmax_T is not None, use Boltzmann exploration over Q-values.
        Otherwise, fall back to epsilon-greedy with parameter eps (with
        bias toward LISTEN via probs = [0.8, 0.1, 0.1]).
        """
        E = obs.shape[0]

        if self.exec_hidden is None or self.exec_hidden[0].shape[1] != E:
            self.init_exec_hidden(E)

        actions = torch.zeros(E, self.n_agents, dtype=torch.long, device=self.device)

        for i in range(self.n_agents):
            q_t, h_new = self.agents[i].forward_step(obs[:, i], self.exec_hidden[i])
            self.exec_hidden[i] = h_new

            if softmax_T is not None:
                prefs = q_t / max(softmax_T, 1e-6)
                probs = torch.softmax(prefs, dim=-1)
                dist = torch.distributions.Categorical(probs)
                sampled_actions = dist.sample()
                actions[:, i] = sampled_actions
            else:
                rand = torch.rand(E, device=self.device)
                explore = rand < eps

                probs = torch.tensor([0.8, 0.1, 0.1], device=self.device)
                random_actions = torch.multinomial(probs, num_samples=E, replacement=True)

                greedy_actions = q_t.argmax(dim=-1)
                actions[:, i] = torch.where(explore, random_actions, greedy_actions)

        return actions

    # ------------------------------------------------------------------
    # Target updates
    # ------------------------------------------------------------------
    @torch.no_grad()
    def update_targets(self, tau: float = 0.01):
        for i in range(self.n_agents):
            for tgt_p, p in zip(self.target_agents[i].parameters(), self.agents[i].parameters()):
                tgt_p.data.copy_(tau * p.data + (1 - tau) * tgt_p.data)
        for tgt_p, p in zip(self.target_mixer.parameters(), self.mixer.parameters()):
            tgt_p.data.copy_(tau * p.data + (1 - tau) * tgt_p.data)
        for tgt_p, p in zip(self.target_central_head.parameters(), self.central_head.parameters()):
            tgt_p.data.copy_(tau * p.data + (1 - tau) * tgt_p.data)

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------
    def train_step(self, replay_buffer, batch_size: int, burn_in: int, learn_len: int):
        """
        R2D2-style training with burn-in + learn window.

        Returns:
            (loss_total, loss_dec, loss_cen) as floats, or None if no update.
        """
        if replay_buffer.num_episodes() == 0:
            return None

        device = self.device
        T = burn_in + learn_len

        batch = replay_buffer.sample_sequences(batch_size, T)

        def to_tensor(x, dtype):
            return torch.tensor(x, dtype=dtype, device=device)

        obs             = to_tensor(batch["obs"], torch.long)
        next_obs        = to_tensor(batch["next_obs"], torch.long)
        state           = to_tensor(batch["state"], torch.float32)
        next_state      = to_tensor(batch["next_state"], torch.float32)
        tau_vec         = to_tensor(batch["tau_vec"], torch.float32)
        next_tau_vec    = to_tensor(batch["next_tau_vec"], torch.float32)
        comm_mask       = to_tensor(batch["comm_mask"], torch.float32)
        next_comm_mask  = to_tensor(batch["next_comm_mask"], torch.float32)
        actions         = to_tensor(batch["actions"], torch.long)
        reward          = to_tensor(batch["reward"], torch.float32)
        done            = to_tensor(batch["done"], torch.float32)

        B, T_actual, _ = obs.shape
        if T_actual < T:
            return None

        learn_start = burn_in
        learn_end = burn_in + learn_len
        K = learn_len

        # 1) ONLINE unroll with hidden sequence (for future central head use)
        chosen_q_seq_list = []
        hidden_seq_list = []
        for i in range(self.n_agents):
            obs_seq_i = obs[:, :, i]
            q_seq_i, h_seq_i, _ = self.agents[i].forward_seq_with_hidden(obs_seq_i)
            a_i = actions[:, :, i].unsqueeze(-1)
            q_i_seq = q_seq_i.gather(2, a_i).squeeze(-1)
            chosen_q_seq_list.append(q_i_seq)
            hidden_seq_list.append(h_seq_i)

        chosen_q_seq = torch.stack(chosen_q_seq_list, dim=-1)          # [B, T, n_agents]
        hidden_seq = torch.stack(hidden_seq_list, dim=2)               # [B, T, n_agents, H]

        chosen_q_learn = chosen_q_seq[:, learn_start:learn_end, :]     # [B, K, n_agents]
        state_learn = state[:, learn_start:learn_end, :]
        tau_vec_learn = tau_vec[:, learn_start:learn_end, :]
        comm_mask_learn = comm_mask[:, learn_start:learn_end, :]
        reward_learn = reward[:, learn_start:learn_end]
        done_learn = done[:, learn_start:learn_end]
        next_state_learn = next_state[:, learn_start:learn_end, :]
        next_tau_vec_learn = next_tau_vec[:, learn_start:learn_end, :]
        next_comm_mask_learn = next_comm_mask[:, learn_start:learn_end, :]
        next_obs_learn = next_obs[:, learn_start:learn_end, :]
        hidden_learn = hidden_seq[:, learn_start:learn_end, :, :]

        BK = B * K
        chosen_q_flat = chosen_q_learn.reshape(BK, self.n_agents)
        state_flat = state_learn.reshape(BK, self.state_dim)
        tau_vec_flat = tau_vec_learn.reshape(BK, self.n_subsets)
        comm_mask_flat = comm_mask_learn.reshape(BK, self.n_agents)
        reward_flat = reward_learn.reshape(BK)
        done_flat = done_learn.reshape(BK)

        # 2) TARGET unroll on next_obs
        with torch.no_grad():
            next_chosen_q_seq_list = []
            for i in range(self.n_agents):
                next_obs_seq_i = next_obs[:, :, i]
                q_next_seq_i, _ = self.target_agents[i].forward_seq(next_obs_seq_i)
                next_a_i = q_next_seq_i.argmax(dim=-1, keepdim=True)
                next_q_i_seq = q_next_seq_i.gather(2, next_a_i).squeeze(-1)
                next_chosen_q_seq_list.append(next_q_i_seq)

            next_chosen_q_seq = torch.stack(next_chosen_q_seq_list, dim=-1)
            next_chosen_q_learn = next_chosen_q_seq[:, learn_start:learn_end, :]

            next_chosen_q_flat = next_chosen_q_learn.reshape(BK, self.n_agents)
            next_state_flat = next_state_learn.reshape(BK, self.state_dim)
            next_tau_vec_flat = next_tau_vec_learn.reshape(BK, self.n_subsets)
            next_comm_mask_flat = next_comm_mask_learn.reshape(BK, self.n_agents)

            q_tot_next_flat = self.target_mixer(
                next_chosen_q_flat,
                next_state_flat,
                next_tau_vec_flat,
                next_comm_mask_flat,
            )

            target_q_tot_flat = reward_flat + self.gamma * (1.0 - done_flat) * q_tot_next_flat

            target_q_dec_flat = (
                reward_flat.unsqueeze(-1)
                + self.gamma * (1.0 - done_flat).unsqueeze(-1) * next_chosen_q_flat
            )

        # 3) TD losses
        loss_dec = self.loss_fn(chosen_q_flat, target_q_dec_flat)

        q_tot_flat = self.mixer(
            chosen_q_flat,
            state_flat,
            tau_vec_flat,
            comm_mask_flat,
        )
        loss_cen = self.loss_fn(q_tot_flat, target_q_tot_flat)

        loss = loss_dec + self.central_loss_coef * loss_cen

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.params, 10.0)
        self.optimizer.step()

        return float(loss.item()), float(loss_dec.item()), float(loss_cen.item())

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    def save_checkpoint(self, path: str, step: int):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            "step": step,
            "agents": [a.state_dict() for a in self.agents],
            "target_agents": [a.state_dict() for a in self.target_agents],
            "mixer": self.mixer.state_dict(),
            "target_mixer": self.target_mixer.state_dict(),
            "central_head": self.central_head.state_dict(),
            "target_central_head": self.target_central_head.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(state, path)

    def load_checkpoint(self, path: str):
        if not os.path.isfile(path):
            print(f"[SD_DRQN] No checkpoint found at {path}")
            return 0
        state = torch.load(path, map_location=self.device)
        for i in range(self.n_agents):
            self.agents[i].load_state_dict(state["agents"][i])
            self.target_agents[i].load_state_dict(state["target_agents"][i])
        self.mixer.load_state_dict(state["mixer"])
        self.target_mixer.load_state_dict(state["target_mixer"])
        if "central_head" in state:
            self.central_head.load_state_dict(state["central_head"])
        if "target_central_head" in state:
            self.target_central_head.load_state_dict(state["target_central_head"])
        self.optimizer.load_state_dict(state["optimizer"])
        print(f"[SD_DRQN] Loaded checkpoint from {path}, step={state['step']}")
        return state["step"]
