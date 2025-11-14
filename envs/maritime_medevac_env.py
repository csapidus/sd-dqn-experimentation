import torch


class MaritimeMedevacVecEnv:
    """
    Simple 2-agent Maritime MEDEVAC environment with vectorized batch dimension.

    Agents:
        0 = helicopter
        1 = boat

    Grid: 4x4, positions are flattened indices 0..15 (x + 4*y)

    Special cells:
        hospital: (0,0)
        pier:     (1,0)
        islands:  (2,2), (3,1)

    Casualties:
        - One casualty initially on each island (so total 2).
        - Boat and helo can carry at most one casualty at a time.
        - Boat can transfer casualty to pier.
        - Helo can pick up from island or pier and deliver to hospital.

    Communication:
        - If Chebyshev distance between helo and boat <= 1, they are "in range".
        - Then comm_mask[e,i] = 1 for both agents, and tau for subset {helo,boat}
          is reset to 0. Otherwise tau increments up to tau_max.
    """

    # Action indices
    ACT_N, ACT_S, ACT_E, ACT_W, ACT_HOLD, ACT_LOAD, ACT_UNLOAD = range(7)

    def __init__(
        self,
        n_envs: int = 1,
        max_steps: int = 20,
        tau_max: int = 10,
        n_agents: int = 2,
        device: torch.device = torch.device("cpu"),
        seed: int = 0,
    ):
        assert n_agents == 2, "This env currently supports exactly 2 agents."
        self.n_envs = n_envs
        self.n_agents = n_agents
        self.max_steps = max_steps
        self.tau_max = tau_max
        self.device = device

        # simple counters for debugging
        self.stats_evacuations = 0
        self.stats_pickups_boat = 0
        self.stats_pickups_helo = 0

        # ----- reward constants -----
        self.r_step = -0.5          # time cost each step
        self.r_pickup_boat = +10.0   # boat picks up from island
        self.r_pickup_helo = +15.0   # helo picks up from island OR boat
        self.r_delivery = +40.0     # patient delivered to hospital
        self.r_invalid = -2.0       # bump for clearly useless actions

        torch.manual_seed(seed)

        # grid
        self.grid_w = 4
        self.grid_h = 4
        self.n_pos = self.grid_w * self.grid_h

        self.hospital_pos = self._xy_to_idx(0, 0)
        self.pier_pos = self._xy_to_idx(1, 0)
        self.island_pos = [self._xy_to_idx(2, 2), self._xy_to_idx(3, 1)]
        self.n_islands = len(self.island_pos)

        # boat allowed everywhere for now
        self.boat_allowed = torch.ones(self.n_pos, dtype=torch.bool)

        # observation space (discrete index 0..1279)
        self.n_obs = 1280

        # state_dim for centralized critic / mixer
        self.state_dim = self.n_pos * 2 + self.n_islands + 1 + 2

        # comm subsets
        self.subset_masks = self._build_subset_masks(n_agents).to(device)
        self.n_subsets = self.subset_masks.shape[0]

        # per-env dynamic state
        self.helo_pos = torch.zeros(n_envs, dtype=torch.long, device=device)
        self.boat_pos = torch.zeros(n_envs, dtype=torch.long, device=device)
        self.steps = torch.zeros(n_envs, dtype=torch.long, device=device)
        self.tau = torch.zeros(n_envs, self.n_subsets, dtype=torch.long, device=device)

        self.island_counts = torch.zeros(
            n_envs, self.n_islands, dtype=torch.long, device=device
        )
        self.pier_count = torch.zeros(n_envs, dtype=torch.long, device=device)
        self.helo_has_patient = torch.zeros(n_envs, dtype=torch.long, device=device)
        self.boat_has_patient = torch.zeros(n_envs, dtype=torch.long, device=device)

        self.n_actions = 7

    # ---------- helpers ----------

    def _xy_to_idx(self, x: int, y: int) -> int:
        return y * self.grid_w + x

    def _idx_to_xy(self, idx: torch.Tensor):
        x = idx % self.grid_w
        y = idx // self.grid_w
        return x, y

    def _build_subset_masks(self, n_agents: int) -> torch.Tensor:
        masks = []
        for mask in range(1, 2**n_agents):
            bits = [(mask >> i) & 1 for i in range(n_agents)]
            masks.append(bits)
        return torch.tensor(masks, dtype=torch.long)

    def _subset_id_from_comm_mask(self, comm_mask: torch.Tensor) -> torch.Tensor:
        """
        comm_mask: [E, n_agents] in {0,1}
        Returns subset_ids: [E] in {0..2^n_agents-1}, where 0 = "no comm".
        """
        E = comm_mask.shape[0]
        subset_ids = torch.zeros(E, dtype=torch.long, device=comm_mask.device)
        for idx, bits in enumerate(self.subset_masks, start=1):
            mask = torch.all(
                comm_mask == bits.view(1, -1).float(), dim=1
            )
            subset_ids[mask] = idx
        return subset_ids

    # ---------- public API ----------

    def reset(self):
        E = self.n_envs
        device = self.device

        # helo at hospital, boat at pier
        self.helo_pos.fill_(self.hospital_pos)
        self.boat_pos.fill_(self.pier_pos)
        self.steps.zero_()
        self.tau.zero_()

        # casualties
        self.island_counts = torch.ones(
            E, self.n_islands, dtype=torch.long, device=device
        )
        self.pier_count.zero_()
        self.helo_has_patient.zero_()
        self.boat_has_patient.zero_()

        obs = self._encode_obs_all()
        state = self._encode_state_all()
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

    def step(self, actions):
        """
        actions: LongTensor [E, 2]   ([:,0]=helo, [:,1]=boat)

        Returns:
            data_dict, reward
        where data_dict has the same keys as reset().
        """
        E = self.n_envs
        device = self.device

        # increment per-env step counters (this is what used to be self.t)
        self.steps += 1

        # base step cost
        reward = torch.full((E,), self.r_step, device=device)

        # snapshot before load/unload
        before_boat_has = self.boat_has_patient.clone()
        before_helo_has = self.helo_has_patient.clone()

        boat_actions = actions[:, 1]
        helo_actions = actions[:, 0]

        self._apply_movement(self.boat_pos, boat_actions, is_boat=True)
        self._apply_movement(self.helo_pos, helo_actions, is_boat=False)

        self._apply_load_unload_boat(boat_actions)
        self._apply_load_unload_helo(helo_actions)

        picked_by_boat = (self.boat_has_patient == 1) & (before_boat_has == 0)
        picked_by_helo = (self.helo_has_patient == 1) & (before_helo_has == 0)
        delivered = (
            (before_helo_has == 1)
            & (self.helo_has_patient == 0)
            & (self.helo_pos == self.hospital_pos)
        )

        invalid_boat = (
            (boat_actions == self.ACT_LOAD) | (boat_actions == self.ACT_UNLOAD)
        ) & ~picked_by_boat
        invalid_helo = (
            (helo_actions == self.ACT_LOAD) | (helo_actions == self.ACT_UNLOAD)
        ) & ~(picked_by_helo | delivered)
        invalid = invalid_boat | invalid_helo

        self.stats_pickups_boat += int(picked_by_boat.sum().item())
        self.stats_pickups_helo += int(picked_by_helo.sum().item())
        self.stats_evacuations += int(delivered.sum().item())

        reward += picked_by_boat.float() * self.r_pickup_boat
        reward += picked_by_helo.float() * self.r_pickup_helo
        reward += delivered.float() * self.r_delivery
        reward += invalid.float() * self.r_invalid

        done = (self.steps >= self.max_steps) | self._all_patients_delivered()

        # obs/state/tau/comm in expected format
        obs, state, tau_vec, comm_mask = self._get_obs_state_tau_comm()

        data = {
            "obs": obs,
            "state": state,
            "tau_vec": tau_vec,
            "comm_mask": comm_mask,
            "done": done.float(),
        }
        return data, reward

    # ---------- internal dynamics helpers ----------

    def _apply_movement(self, pos: torch.Tensor, actions: torch.Tensor, is_boat: bool):
        E = pos.shape[0]
        delta = torch.zeros(E, dtype=torch.long, device=pos.device)

        delta[actions == self.ACT_N] = -self.grid_w
        delta[actions == self.ACT_S] = self.grid_w
        delta[actions == self.ACT_W] = -1
        delta[actions == self.ACT_E] = 1

        new_pos = pos + delta

        # check grid bounds
        x, y = self._idx_to_xy(new_pos)
        valid = (x >= 0) & (x < self.grid_w) & (y >= 0) & (y < self.grid_h)

        if is_boat:
            boat_valid = torch.ones_like(valid)
            valid_idx = torch.nonzero(valid, as_tuple=False).squeeze(-1)
            if valid_idx.numel() > 0:
                boat_valid[valid_idx] = self.boat_allowed[
                    new_pos[valid_idx].cpu()
                ]
            valid = valid & boat_valid

        pos[:] = torch.where(valid, new_pos, pos)

    def _apply_load_unload_boat(self, boat_actions: torch.Tensor):
        load_mask = boat_actions == self.ACT_LOAD
        if load_mask.any():
            idx = torch.nonzero(load_mask, as_tuple=False).squeeze(-1)
            for e in idx.tolist():
                if self.boat_has_patient[e] == 1:
                    continue
                pos = int(self.boat_pos[e].item())
                if pos in self.island_pos:
                    island_idx = self.island_pos.index(pos)
                    if self.island_counts[e, island_idx] > 0:
                        self.island_counts[e, island_idx] -= 1
                        self.boat_has_patient[e] = 1

        unload_mask = boat_actions == self.ACT_UNLOAD
        if unload_mask.any():
            idx = torch.nonzero(unload_mask, as_tuple=False).squeeze(-1)
            for e in idx.tolist():
                if self.boat_has_patient[e] == 0:
                    continue
                pos = int(self.boat_pos[e].item())
                if pos == self.pier_pos:
                    self.boat_has_patient[e] = 0
                    self.pier_count[e] += 1

    def _apply_load_unload_helo(self, helo_actions: torch.Tensor):
        load_mask = helo_actions == self.ACT_LOAD
        if load_mask.any():
            idx = torch.nonzero(load_mask, as_tuple=False).squeeze(-1)
            for e in idx.tolist():
                if self.helo_has_patient[e] == 1:
                    continue
                pos = int(self.helo_pos[e].item())
                if pos in self.island_pos:
                    island_idx = self.island_pos.index(pos)
                    if self.island_counts[e, island_idx] > 0:
                        self.island_counts[e, island_idx] -= 1
                        self.helo_has_patient[e] = 1
                elif pos == self.pier_pos and self.pier_count[e] > 0:
                    self.pier_count[e] -= 1
                    self.helo_has_patient[e] = 1

        unload_mask = helo_actions == self.ACT_UNLOAD
        if unload_mask.any():
            idx = torch.nonzero(unload_mask, as_tuple=False).squeeze(-1)
            for e in idx.tolist():
                if self.helo_has_patient[e] == 0:
                    continue
                pos = int(self.helo_pos[e].item())
                if pos == self.hospital_pos:
                    self.helo_has_patient[e] = 0

    # ---------- encoding ----------

    def _encode_obs_all(self) -> torch.Tensor:
        E = self.n_envs
        obs = torch.zeros(E, self.n_agents, dtype=torch.long, device=self.device)
        for e in range(E):
            obs[e, 0] = self._encode_obs_single(e, agent_id=0)
            obs[e, 1] = self._encode_obs_single(e, agent_id=1)
        return obs

    def _encode_obs_single(self, e: int, agent_id: int) -> int:
        """
        Local observation -> single discrete index.
        """
        if agent_id == 0:
            pos = int(self.helo_pos[e].item())
            has_patient = int(self.helo_has_patient[e].item())
            other_pos = int(self.boat_pos[e].item())
        else:
            pos = int(self.boat_pos[e].item())
            has_patient = int(self.boat_has_patient[e].item())
            other_pos = int(self.helo_pos[e].item())

        if pos == self.hospital_pos:
            region_type = 0
        elif pos == self.pier_pos:
            region_type = 1
        elif pos in self.island_pos:
            region_type = 2
        else:
            region_type = 3

        num_here_flag = 0
        if region_type == 1:
            if self.pier_count[e] > 0:
                num_here_flag = 1
        elif region_type == 2:
            island_idx = self.island_pos.index(pos)
            if self.island_counts[e, island_idx] > 0:
                num_here_flag = 1

        px, py = self._idx_to_xy(torch.tensor(pos, device=self.device))
        qx, qy = self._idx_to_xy(torch.tensor(self.pier_pos, device=self.device))
        dist_pier = int((px - qx).abs().item() + (py - qy).abs().item())
        dist_pier_bin = min(dist_pier, 3)

        ox, oy = self._idx_to_xy(torch.tensor(other_pos, device=self.device))
        dist_other = int((px - ox).abs().item() + (py - oy).abs().item())
        dist_other_bin = dist_other if dist_other <= 3 else 4

        tau_vec = (self.tau[e].clamp(0, self.tau_max).float()) / float(self.tau_max)
        tau_last = float(tau_vec[-1].item())
        tau_bucket = min(int(tau_last * 4.0), 3)

        idx = region_type
        idx = idx * 2 + has_patient
        idx = idx * 2 + num_here_flag
        idx = idx * 4 + dist_pier_bin
        idx = idx * 5 + dist_other_bin
        idx = idx * 4 + tau_bucket
        return idx

    def _encode_state_all(self) -> torch.Tensor:
        E = self.n_envs
        device = self.device
        state = torch.zeros(E, self.state_dim, dtype=torch.float32, device=device)

        for e in range(E):
            offset = 0
            state[e, offset + self.helo_pos[e]] = 1.0
            offset += self.n_pos
            state[e, offset + self.boat_pos[e]] = 1.0
            offset += self.n_pos
            for k in range(self.n_islands):
                state[e, offset + k] = float(self.island_counts[e, k].item()) / 2.0
            offset += self.n_islands
            state[e, offset] = float(self.pier_count[e].item()) / 2.0
            offset += 1
            state[e, offset] = float(self.helo_has_patient[e].item())
            state[e, offset + 1] = float(self.boat_has_patient[e].item())

        return state

    def _chebyshev_distance(self, a_pos: torch.Tensor, b_pos: torch.Tensor) -> torch.Tensor:
        ax, ay = self._idx_to_xy(a_pos)
        bx, by = self._idx_to_xy(b_pos)
        return torch.max((ax - bx).abs(), (ay - by).abs())

    def _get_obs_state_tau_comm(self):
        E = self.n_envs
        device = self.device

        dist = self._chebyshev_distance(self.helo_pos, self.boat_pos)
        in_range = dist <= 1

        comm_mask = torch.zeros(E, self.n_agents, dtype=torch.float32, device=device)
        comm_mask[in_range, :] = 1.0

        subset_ids = self._subset_id_from_comm_mask(comm_mask)

        self.tau = (self.tau + 1).clamp(max=self.tau_max)

        active = subset_ids > 0
        if active.any():
            env_idx = torch.nonzero(active, as_tuple=False).squeeze(-1)
            self.tau[env_idx, subset_ids[env_idx] - 1] = 0

        tau_vec = (self.tau.clamp(0, self.tau_max).float()) / float(self.tau_max)

        obs = self._encode_obs_all()
        state = self._encode_state_all()
        return obs, state, tau_vec, comm_mask

    def _all_patients_delivered(self):
        no_islands = self.island_counts.sum(dim=1) == 0
        no_pier = self.pier_count == 0
        no_helo = self.helo_has_patient == 0
        no_boat = self.boat_has_patient == 0
        return no_islands & no_pier & no_helo & no_boat
