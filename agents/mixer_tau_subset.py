import torch
import torch.nn as nn
import torch.nn.functional as F


class TauSubsetGatedMixer(nn.Module):
    """
    τ-gated mixer that blends decentralized and centralized value
    based on per-subset τ and the active communication subset.

    Inputs:
        chosen_q:  [B, n_agents]      # Q_i(o_i, a_i) for each agent
        state:     [B, state_dim]
        tau_vec:   [B, n_subsets]     # per-subset τ, normalized to [0,1]
        comm_mask: [B, n_agents] in {0,1} (which agents are in current subset)

    Output:
        q_tot:     [B] scalar joint Q

    Intuition:
        - Define an "effective τ" for the active communication subset.
        - When τ_eff ≈ 0  (just communicated)   => rely on centralized mix.
        - When τ_eff ≈ 1  (long since comm)     => rely on decentralized sum.
    """

    def __init__(self, n_agents, state_dim, n_subsets, hidden_dim=128):
        super().__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.n_subsets = n_subsets

        input_dim = n_agents + state_dim + n_agents + n_subsets
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mix_fc = nn.Linear(hidden_dim, 1)

    def forward(self, chosen_q, state, tau_vec, comm_mask):
        """
        chosen_q:  [B, n_agents]
        state:     [B, state_dim]
        tau_vec:   [B, n_subsets]
        comm_mask:[B, n_agents]
        """
        B = chosen_q.size(0)
        device = chosen_q.device

        # Centralized candidate Q based on joint features
        x = torch.cat([chosen_q, state, comm_mask, tau_vec], dim=-1)  # [B, ...]
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        q_cen = self.mix_fc(h).squeeze(-1)          # [B]

        # Decentralized candidate Q: sum of per-agent values
        q_dec = chosen_q.sum(dim=-1)                # [B]

        # ----- τ-based gating -----
        # Reconstruct subset id from comm_mask (same mapping as in env):
        # bitmask 1..2^n_agents-1; 0 means "no communication".
        powers = (2 ** torch.arange(self.n_agents, device=device)).view(1, -1)
        mask_int = (comm_mask.long() * powers).sum(dim=-1)  # [B], 0..2^n_agents-1

        tau_eff = torch.ones(B, device=device)  # default: no comm => fully decentralized
        nonzero = mask_int > 0
        if nonzero.any():
            subset_idx = mask_int[nonzero] - 1             # 0-based index into tau_vec
            tau_eff[nonzero] = tau_vec[nonzero, subset_idx]

        # τ is normalized in [0,1]. Define weights:
        #   w_cen = 1 - τ_eff   (recent comm -> centralized)
        #   w_dec = τ_eff       (long since comm -> decentralized)
        w_cen = 1.0 - tau_eff
        w_dec = tau_eff

        q_tot = w_cen * q_cen + w_dec * q_dec            # [B]
        return q_tot
