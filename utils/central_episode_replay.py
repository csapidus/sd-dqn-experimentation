from collections import deque, namedtuple
import random

CentralStep = namedtuple(
    "CentralStep",
    (
        "obs",        # scalar joint obs index
        "actions",    # scalar joint action index
        "reward",     # scalar
        "next_obs",   # scalar joint obs index
        "done",       # 0/1
    ),
)


class CentralEpisodeReplayBuffer:
    def __init__(self, capacity_episodes=10000):
        self.capacity = capacity_episodes
        self.buffer = deque(maxlen=capacity_episodes)

    def push_episode(self, episode_steps):
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
                ep = max(episodes, key=len)
                seq = ep[:]  # full episode

            batch.append(seq)

        def collect(attr):
            return [[getattr(step, attr) for step in seq] for seq in batch]

        data = {
            "obs": collect("obs"),
            "actions": collect("actions"),
            "reward": collect("reward"),
            "next_obs": collect("next_obs"),
            "done": collect("done"),
        }
        return data
