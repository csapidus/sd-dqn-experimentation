from collections import deque, namedtuple
import random

Step = namedtuple(
    "Step",
    (
        "obs",           # [n_agents]
        "state",         # [state_dim]
        "tau_vec",       # [n_subsets]
        "comm_mask",     # [n_agents]
        "actions",       # [n_agents]
        "reward",        # scalar
        "next_obs",      # [n_agents]
        "next_state",    # [state_dim]
        "next_tau_vec",  # [n_subsets]
        "next_comm_mask",# [n_agents]
        "done",          # scalar 0 or 1
    ),
)


class EpisodeReplayBuffer:
    def __init__(self, capacity_episodes=10000):
        self.capacity = capacity_episodes
        self.buffer = deque(maxlen=capacity_episodes)

    def push_episode(self, episode_steps):
        """episode_steps: list[Step]"""
        if len(episode_steps) > 0:
            self.buffer.append(episode_steps)

    def __len__(self):
        return sum(len(ep) for ep in self.buffer)

    def num_episodes(self):
        return len(self.buffer)

    def sample_sequences(self, batch_size, seq_len):
        episodes = self.buffer
        assert len(episodes) > 0, "No episodes in buffer"

        batch = []

        for _ in range(batch_size):
            valid_eps = [ep for ep in episodes if len(ep) >= seq_len]

            if valid_eps:
                ep = random.choice(valid_eps)
                max_start = len(ep) - seq_len
                start = random.randint(0, max_start)
                seq = ep[start:start + seq_len]
            else:
                # Fall back: use the longest episode we have (may be < seq_len)
                ep = max(episodes, key=len)
                seq = ep[:]  # full episode

            batch.append(seq)

        def collect(attr):
            return [[getattr(step, attr) for step in seq] for seq in batch]

        data = {
            "obs": collect("obs"),
            "state": collect("state"),
            "tau_vec": collect("tau_vec"),
            "comm_mask": collect("comm_mask"),
            "actions": collect("actions"),
            "reward": collect("reward"),
            "next_obs": collect("next_obs"),
            "next_state": collect("next_state"),
            "next_tau_vec": collect("next_tau_vec"),
            "next_comm_mask": collect("next_comm_mask"),
            "done": collect("done"),
        }
        return data

