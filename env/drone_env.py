import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv

class DroneSwarmEnv(MultiAgentEnv):
    def __init__(self, config):
        self.cfg = config
        self.n = config["env"]["num_drones"]
        self.r = config["env"]["num_radars"]
        self.map_size = config["env"]["map_size"]
        self.dt = config["physics"]["dt"]

        self.agents = [f"drone_{i}" for i in range(self.n)]
        self.reset()

    def reset(self, *, seed=None, options=None):
        self.positions = {a: np.random.rand(2)*self.map_size for a in self.agents}
        self.velocities = {a: np.zeros(2) for a in self.agents}

        self.radars = [{
            "pos": np.random.rand(2)*self.map_size,
            "alive": True
        } for _ in range(self.r)]

        return self._obs(), {}

    def step(self, actions):
        rewards = {}

        for a, act in actions.items():
            accel = self._action_to_accel(act)
            vel = self.velocities[a]

            vel += accel * self.dt
            vel *= (1 - self.cfg["physics"]["drag_coeff"])
            self.positions[a] += vel * self.dt

            self.velocities[a] = vel

            reward = 1.0
            reward += self._interact_radars(a, act)

            rewards[a] = reward

        done = all(not r["alive"] for r in self.radars)

        return self._obs(), rewards, {"__all__": done}, {"__all__": False}, {}

    def _action_to_accel(self, act):
        mapping = {
            0: np.array([1,0]),
            1: np.array([-1,0]),
            2: np.array([0,1]),
            3: np.array([0,-1]),
            4: np.array([0,0])
        }
        return mapping.get(act, np.zeros(2))

    def _interact_radars(self, agent, act):
        pos = self.positions[agent]
        reward = 0

        for r in self.radars:
            dist = np.linalg.norm(pos - r["pos"])
            if act == 4 and dist < 10 and r["alive"]:
                r["alive"] = False
                reward += 20
            if act == 3 and dist < 20:
                reward += 10

        return reward

    def _obs(self):
        obs = {}
        for a in self.agents:
            pos = self.positions[a] / self.map_size
            vel = self.velocities[a]

            neighbors, mask = [], []
            for b in self.agents:
                if b == a:
                    continue
                dist = np.linalg.norm(self.positions[a] - self.positions[b])
                if dist < self.cfg["communication"]["comm_radius"]:
                    neighbors.append(np.concatenate([self.positions[b]/self.map_size, self.velocities[b]]))
                    mask.append(1)
                else:
                    neighbors.append(np.zeros(4))
                    mask.append(0)

            obs[a] = {
                "self": np.concatenate([pos, vel]).astype(np.float32),
                "neighbors": np.array(neighbors, dtype=np.float32),
                "mask": np.array(mask, dtype=np.float32)
            }
        return obs