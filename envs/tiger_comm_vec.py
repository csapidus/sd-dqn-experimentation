import torch

class TigerCommVecEnv:
    """Vectorized 2-agent Dec-Tiger environment with communication subsets
    and per-subset tau (time since last comm event).

    Hidden state per env:
        tiger_pos ∈ {0, 1}   # 0=LEFT, 1=RIGHT

    Actions per agent:
        0 = LISTEN
        1 = OPEN_LEFT
        2 = OPEN_RIGHT

    Observations per agent:
        0 = "hear_left"
        1 = "hear_right"
        2 = "no_observation" (if opened a door or at reset)

    Communication mask per env:
        comm_mask[e, i] in {0,1} indicates whether agent i is in the current
        communicating subset at this step.

    Subsets:
        For n_agents, we consider all non-empty subsets.
        There are n_subsets = 2^n_agents - 1 possible non-empty subsets, each
        represented by a bitmask in [1..2^n_agents-1].

    Tau:
        tau[e, s] tracks time since last comm event for subset s (0-based index).
        At each step:
            - All tau[e, :] are incremented by 1 (clipped at tau_max).
            - For the active subset id (if any), tau[e, active_subset_idx] is reset to 0.
    """

    def __init__(
        self,
        n_envs: int,
        listen_correct_prob: float = 0.85,
        max_steps: int = 20,
        tau_max: int = 10,
        n_agents: int = 2,
        device: str = "cpu",
        seed: int = 0,
        eta: float = 3.0,
    ):
        self.n_envs = n_envs
        self.n_agents = n_agents
        self.n_actions = 3  # listen, open L, open R
        self.n_obs = 3      # hear L, hear R, no obs
        self.state_dim = 2  # one-hot for tiger left/right
        self.listen_correct_prob = listen_correct_prob
        self.max_steps = max_steps
        self.tau_max = tau_max
        self.device = device
        self.eta = eta

        torch.manual_seed(seed)

        # All non-empty subsets of agents: bitmask 1..(2^n_agents-1)
        self.subset_masks = self._build_subset_masks(n_agents).to(device)
        self.n_subsets = self.subset_masks.shape[0]

        # Per-env hidden state
        self.tiger_pos = torch.zeros(n_envs, dtype=torch.long, device=device)
        self.steps = torch.zeros(n_envs, dtype=torch.long, device=device)
        self.tau = torch.zeros(n_envs, self.n_subsets, dtype=torch.long, device=device)

        self.belief = torch.full(
            (n_envs,), 0.5, dtype=torch.float32, device=device
        )

    def _build_subset_masks(self, n_agents: int) -> torch.Tensor:
        masks = []
        for mask in range(1, 2 ** n_agents):
            bits = [(mask >> i) & 1 for i in range(n_agents)]
            masks.append(bits)
        return torch.tensor(masks, dtype=torch.long)

    def _subset_id_from_comm_mask(self, comm_mask: torch.Tensor) -> torch.Tensor:
        E = comm_mask.shape[0]
        device = comm_mask.device
        powers = (2 ** torch.arange(self.n_agents, device=device)).view(1, -1)
        mask_int = (comm_mask.long() * powers).sum(dim=-1)  # [E]
        return mask_int  # 0..2^n_agents-1

    def _entropy(self, p: torch.Tensor) -> torch.Tensor:
        # Binary entropy H(p) = -p log p - (1-p) log (1-p)
        return -p * torch.log(p + 1e-9) - (1.0 - p) * torch.log(1.0 - p + 1e-9)

    def reset(self):
        E = self.n_envs
        device = self.device

        # Sample tiger positions
        self.tiger_pos = torch.randint(low=0, high=2, size=(E,), device=device)
        self.steps = torch.zeros(E, dtype=torch.long, device=device)
        self.tau = torch.zeros(E, self.n_subsets, dtype=torch.long, device=device)

        # NEW: reset belief to uninformative prior
        self.belief = torch.full(
            (E,), 0.5, dtype=torch.float32, device=device
        )

        # At reset, obs are NO_OBS (2)
        obs = torch.full((E, self.n_agents), 2, dtype=torch.long, device=device)

        # State one-hot
        state = torch.zeros(E, self.state_dim, dtype=torch.float32, device=device)
        state[torch.arange(E, device=device), self.tiger_pos] = 1.0

        tau_norm = (self.tau.clamp(0, self.tau_max).float()) / float(self.tau_max)
        comm_mask = torch.zeros(E, self.n_agents, dtype=torch.float32, device=device)
        done = torch.zeros(E, dtype=torch.float32, device=device)

        return {
            "obs": obs,
            "state": state,
            "tau_vec": tau_norm,
            "comm_mask": comm_mask,
            "done": done,
        }

    def step(self, actions: torch.Tensor):
        device = self.device
        E = self.n_envs
        assert actions.shape == (E, self.n_agents)

        # Increment steps
        self.steps += 1

        # Decode actions
        listen = actions == 0
        open_left = actions == 1
        open_right = actions == 2
        any_open = (open_left | open_right).any(dim=-1)

        # Tiger door: 0=left, 1=right
        tiger_left = self.tiger_pos == 0
        tiger_right = self.tiger_pos == 1

        # ------------------------------------------------------
        # Rewards for Dec-Tiger (joint team reward), Table 2.1
        #
        # Actions: a_Li = 0, a_OL = 1, a_OR = 2
        # joint action  |  s_l (tiger left) |  s_r (tiger right)
        # <Li,Li>       |        -2         |        -2
        # <Li,OL>       |       -101        |        +9
        # <Li,OR>       |        +9         |       -101
        # <OL,Li>       |       -101        |        +9
        # <OL,OL>       |        -50        |        +20
        # <OL,OR>       |       -100        |       -100
        # <OR,Li>       |        +9         |       -101
        # <OR,OL>       |       -100        |       -100
        # <OR,OR>       |        +20        |        -50
        # ------------------------------------------------------
        reward = torch.zeros(E, dtype=torch.float32, device=device)

        # cache prior belief for entropy bonus
        prior_belief = self.belief.clone()

        a0 = actions[:, 0]
        a1 = actions[:, 1]

        li0 = (a0 == 0)
        ol0 = (a0 == 1)
        or0 = (a0 == 2)

        li1 = (a1 == 0)
        ol1 = (a1 == 1)
        or1 = (a1 == 2)

        # <Li,Li>
        mask = li0 & li1
        reward[mask] = -2.0

        # <Li,OL>
        mask = li0 & ol1 & tiger_left
        reward[mask] = -101.0
        mask = li0 & ol1 & tiger_right
        reward[mask] = +9.0

        # <Li,OR>
        mask = li0 & or1 & tiger_left
        reward[mask] = +9.0
        mask = li0 & or1 & tiger_right
        reward[mask] = -101.0

        # <OL,Li>
        mask = ol0 & li1 & tiger_left
        reward[mask] = -101.0
        mask = ol0 & li1 & tiger_right
        reward[mask] = +9.0

        # <OL,OL>
        mask = ol0 & ol1 & tiger_left
        reward[mask] = -50.0
        mask = ol0 & ol1 & tiger_right
        reward[mask] = +20.0

        # <OL,OR> : -100 regardless of state
        mask = ol0 & or1
        reward[mask] = -100.0

        # <OR,Li>
        mask = or0 & li1 & tiger_left
        reward[mask] = +9.0
        mask = or0 & li1 & tiger_right
        reward[mask] = -101.0

        # <OR,OL> : -100 regardless of state
        mask = or0 & ol1
        reward[mask] = -100.0

        # <OR,OR>
        mask = or0 & or1 & tiger_left
        reward[mask] = +20.0
        mask = or0 & or1 & tiger_right
        reward[mask] = -50.0

        # DEBUG
        # any_open_left = open_left.any(dim=-1)
        # any_open_right = open_right.any(dim=-1)

        # opened_tiger = (any_open_left & tiger_left) | (any_open_right & tiger_right)
        # opened_gold = (any_open_left & tiger_right) | (any_open_right & tiger_left)

        # For sanity test central run
        # Small negative per-step cost, asymmetric open rewards
        # step_penalty = -0.01   # or -0.05

        # reward = torch.full((E,), step_penalty, dtype=torch.float32, device=device)
        # reward = reward + opened_tiger.float() * (-1.0) + opened_gold.float() * 3.0

        # Done if any_open or max_steps reached
        done = torch.zeros(E, dtype=torch.bool, device=device)
        done = done | any_open | (self.steps >= self.max_steps)

        # Generate observations
        obs = torch.full((E, self.n_agents), 2, dtype=torch.long, device=device)
        not_done = ~done
        if not_done.any():
            idx_env = torch.nonzero(not_done, as_tuple=False).squeeze(-1)
            tiger_pos_env = self.tiger_pos[idx_env]  # [n_active]
            correct = tiger_pos_env
            incorrect = 1 - tiger_pos_env

            listen_active = listen[idx_env]
            n_active = idx_env.shape[0]
            rand = torch.rand(n_active, self.n_agents, device=device)
            correct_mask = rand < self.listen_correct_prob

            obs_active = torch.full(
                (n_active, self.n_agents), 2, dtype=torch.long, device=device
            )
            for i in range(self.n_agents):
                listen_i = listen_active[:, i]
                if listen_i.any():
                    idx_i = torch.nonzero(listen_i, as_tuple=False).squeeze(-1)
                    ci = correct[idx_i]
                    ii = incorrect[idx_i]
                    cm = correct_mask[idx_i, i]
                    oi = torch.where(cm, ci, ii)
                    obs_active[idx_i, i] = oi

            obs[idx_env] = obs_active

        # ------------------------------------------------------
        # Belief update (Bayes) + entropy reduction bonus
        # ------------------------------------------------------
        #  only update belief in envs where BOTH agents listened and the
        # episode has not yet terminated
        listened_both = (listen.sum(dim=-1) == self.n_agents) & (~done)

        if listened_both.any():
            idx = torch.nonzero(listened_both, as_tuple=False).squeeze(-1)
            # use agent 0's observation (both see sync'd observations when listening)
            obs0 = obs[idx, 0]  # 0 = hear_left, 1 = hear_right, 2 = no_obs

            # only update when we have an actual left/right observation
            valid = (obs0 == 0) | (obs0 == 1)
            if valid.any():
                idx2 = idx[valid]
                obs_v = obs0[valid]

                p = self.listen_correct_prob
                # likelihood P(o | tiger=LEFT)
                lik_left = torch.where(obs_v == 0, p, 1.0 - p)
                # likelihood P(o | tiger=RIGHT)
                lik_right = torch.where(obs_v == 1, p, 1.0 - p)

                prior = prior_belief[idx2]
                unnorm_left = prior * lik_left
                unnorm_right = (1.0 - prior) * lik_right
                new_b = unnorm_left / (unnorm_left + unnorm_right + 1e-9)

                self.belief[idx2] = new_b

        # entropy reduction bonus for all envs
        ent_before = self._entropy(prior_belief)
        ent_after = self._entropy(self.belief)
        bonus = self.eta * (ent_before - ent_after)
        # only positive information gain rewarded
        bonus = torch.clamp(bonus, min=0.0, max=5.0)
        reward += bonus

        # DEBUG: fully observable version — leak tiger_pos to all agents
        # obs = self.tiger_pos.view(E, 1).repeat(1, self.n_agents)

        # Update tau: increment all, reset for active subsets (comm_mask derived from listen)
        self.tau = torch.clamp(self.tau + 1, max=self.tau_max)

        comm_mask = listen.float()
        comm_mask[done] = 0.0

        subset_ids = self._subset_id_from_comm_mask(comm_mask)
        nonzero_mask = subset_ids > 0
        if nonzero_mask.any():
            idx_env = torch.nonzero(nonzero_mask, as_tuple=False).squeeze(-1)
            subset_vals = subset_ids[idx_env]
            subset_idx = subset_vals - 1  # our subsets are 1..2^n_agents-1
            self.tau[idx_env, subset_idx] = 0

        # Build state one-hot
        state = torch.zeros(E, self.state_dim, dtype=torch.float32, device=device)
        state[torch.arange(E, device=device), self.tiger_pos] = 1.0

        tau_norm = (self.tau.clamp(0, self.tau_max).float()) / float(self.tau_max)

        return (
            {
                "obs": obs,
                "state": state,
                "tau_vec": tau_norm,
                "comm_mask": comm_mask,
                "done": done.float(),
            },
            reward,
        )