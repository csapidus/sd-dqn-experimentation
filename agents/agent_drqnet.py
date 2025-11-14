import torch
import torch.nn as nn
import torch.nn.functional as F


class AgentDRQNet(nn.Module):
    """Per-agent DRQN (GRU-based Q-network).

    - For training, use forward_seq over sequences [B, T].
    - For execution, use forward_step with hidden state.
    """

    def __init__(self, n_obs, n_actions, hidden_dim=64):
        super().__init__()
        self.n_obs = n_obs
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim

        # treat observations as discrete indices and embed them.
        self.embed = nn.Embedding(n_obs, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, n_actions)

    def forward_seq(self, obs_seq, h0=None):
        """Unroll the DRQN over a whole sequence.

        obs_seq: LongTensor [B, T]
        h0:      FloatTensor [1, B, H] or None

        returns:
            q_seq: [B, T, n_actions]
            hT:    [1, B, H]
        """
        x = self.embed(obs_seq)  # [B, T, H]
        if h0 is None:
            B = obs_seq.shape[0]
            h0 = torch.zeros(1, B, self.hidden_dim, device=obs_seq.device)
        out, hT = self.gru(x, h0)  # out: [B, T, H]
        q_seq = self.fc(out)       # [B, T, A]
        return q_seq, hT

    def forward_seq_with_hidden(self, obs_seq, h0=None):
        """Like forward_seq, but also returns the full hidden sequence.

        Useful fo rbuilding a joint hidden representation for centralized critics or mixers.

        obs_seq: LongTensor [B, T]
        h0:      FloatTensor [1, B, H] or None

        returns:
            q_seq: [B, T, n_actions]
            h_seq: [B, T, H]
            hT:    [1, B, H]
        """
        x = self.embed(obs_seq)  # [B, T, H]
        if h0 is None:
            B = obs_seq.shape[0]
            h0 = torch.zeros(1, B, self.hidden_dim, device=obs_seq.device)
        out, hT = self.gru(x, h0)  # out: [B, T, H]
        q_seq = self.fc(out)       # [B, T, A]
        return q_seq, out, hT

    def forward_step(self, obs_t, h_prev=None):
        """Single-step forward for execution.

        obs_t:  LongTensor [E]
        h_prev: FloatTensor [1, E, H] or None

        returns:
            q_t:   [E, n_actions]
            h_new: [1, E, H]
        """
        E = obs_t.shape[0]
        x = self.embed(obs_t)             # [E, H]
        x = x.unsqueeze(1)                # [E, 1, H]
        if h_prev is None:
            h_prev = torch.zeros(1, E, self.hidden_dim, device=obs_t.device)
        out, h_new = self.gru(x, h_prev)  # out: [E, 1, H]
        out = out.squeeze(1)              # [E, H]
        q_t = self.fc(out)                # [E, A]
        return q_t, h_new
