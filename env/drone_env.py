import numpy as np
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv

class DroneSwarmEnv(MultiAgentEnv):
    def __init__(self, config):
        self.cfg = config
        self.n = config["env"]["num_drones"]
        self.r = config["env"]["num_radars"]
        self.map_size = config["env"]["map_size"]
        self.dt = config["physics"]["dt"]
        self.max_accel = config["physics"]["max_acceleration"]

        self.agents = [f"drone_{i}" for i in range(self.n)]
        self.possible_agents = self.agents[:]
        self.max_steps = config["env"]["max_steps"]
        self._step_count = 0

        self.comm_radius = config["communication"]["comm_radius"]
        self.radar_speed = 1.0
        self.radar_detect_range = 15.0
        self.grid_cell_size = 10.0
        self.intrinsic_weight = 0.01

        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "self": spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float32),
            "neighbors": spaces.Box(-np.inf, np.inf, shape=(self.n - 1, 4), dtype=np.float32),
            "mask": spaces.Box(0, 1, shape=(self.n - 1,), dtype=np.float32),
            "global_state": spaces.Box(-np.inf, np.inf, shape=(self.n * 4,), dtype=np.float32),
        })

        self.visit_counts = {}
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self._step_count = 0
        self.positions = {a: np.random.rand(2) * self.map_size for a in self.agents}
        self.velocities = {a: np.zeros(2) for a in self.agents}

        self.radars = [{
            "pos": np.random.rand(2) * self.map_size,
            "vel": np.random.randn(2) * self.radar_speed * 0.5,
            "alive": True,
        } for _ in range(self.r)]

        self.visit_counts = {}
        return self._obs(), {}

    def step(self, actions):
        self._step_count += 1
        rewards = {}

        for a, act in actions.items():
            accel = np.clip(act, -1.0, 1.0) * self.max_accel
            vel = self.velocities[a]

            vel += accel * self.dt
            vel *= (1 - self.cfg["physics"]["drag_coeff"])
            self.positions[a] += vel * self.dt
            self.positions[a] = np.clip(self.positions[a], 0, self.map_size)

            self.velocities[a] = vel

            reward = 1.0
            reward += self._interact_radars(a, accel)
            reward += self._intrinsic_reward(a)

            rewards[a] = reward

        self._move_radars()

        all_destroyed = all(not r["alive"] for r in self.radars)
        done = all_destroyed or self._step_count >= self.max_steps
        terminations = {a: done for a in self.agents}
        terminations["__all__"] = done

        return self._obs(), rewards, terminations, {"__all__": False}, {}

    def _interact_radars(self, agent, act):
        pos = self.positions[agent]
        act_mag = np.linalg.norm(act)
        reward = 0

        for r in self.radars:
            if not r["alive"]:
                continue
            dist = np.linalg.norm(pos - r["pos"])

            if dist < 20:
                reward += (20 - dist) / 20 * 5

            if dist < 5 and act_mag < 0.2:
                r["alive"] = False
                reward += 20

        return reward

    def _move_radars(self):
        for r in self.radars:
            if not r["alive"]:
                continue

            nearest_dist = float("inf")
            nearest_pos = None
            for a in self.agents:
                d = np.linalg.norm(self.positions[a] - r["pos"])
                if d < nearest_dist:
                    nearest_dist = d
                    nearest_pos = self.positions[a]

            if nearest_dist < self.radar_detect_range and nearest_pos is not None:
                direction = (r["pos"] - nearest_pos) / (nearest_dist + 1e-8)
                r["vel"] += direction * self.radar_speed * self.dt * 0.5
            else:
                r["vel"] += np.random.randn(2) * self.radar_speed * self.dt * 0.3

            r["vel"] = np.clip(r["vel"], -self.radar_speed, self.radar_speed)
            r["pos"] += r["vel"] * self.dt
            r["pos"] = np.clip(r["pos"], 0, self.map_size)

    def _intrinsic_reward(self, agent):
        cell = tuple(np.floor(self.positions[agent] / self.grid_cell_size).astype(int))
        count = self.visit_counts.get(cell, 0)
        self.visit_counts[cell] = count + 1
        return 1.0 / np.sqrt(count + 1) * self.intrinsic_weight

    def _obs(self):
        obs = {}
        global_parts = []
        for a in self.agents:
            pos = self.positions[a] / self.map_size
            vel = self.velocities[a]
            global_parts.extend([pos[0], pos[1], vel[0], vel[1]])

        global_state = np.array(global_parts, dtype=np.float32)

        for a in self.agents:
            pos = self.positions[a] / self.map_size
            vel = self.velocities[a]

            neighbors, mask = [], []
            for b in self.agents:
                if b == a:
                    continue
                dist = np.linalg.norm(self.positions[a] - self.positions[b])
                if dist < self.comm_radius:
                    neighbors.append(np.concatenate([self.positions[b] / self.map_size, self.velocities[b]]))
                    mask.append(1)
                else:
                    neighbors.append(np.zeros(4))
                    mask.append(0)

            obs[a] = {
                "self": np.concatenate([pos, vel]).astype(np.float32),
                "neighbors": np.array(neighbors, dtype=np.float32),
                "mask": np.array(mask, dtype=np.float32),
                "global_state": global_state,
            }
        return obs
