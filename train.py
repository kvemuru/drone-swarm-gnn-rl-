import ray
import yaml
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog

from env.drone_env import DroneSwarmEnv
from models.gnn_mappo_model import GNNMAPPOModel

ModelCatalog.register_custom_model("gnn_mappo", GNNMAPPOModel)

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

ray.init()

config = (
    PPOConfig()
    .environment(env=DroneSwarmEnv, env_config=cfg)
    .framework("torch")
    .rollouts(num_rollout_workers=cfg["training"]["num_workers"])
    .training(
        model={"custom_model": "gnn_mappo"},
        train_batch_size=cfg["training"]["train_batch_size"],
        gamma=cfg["training"]["gamma"]
    )
    .multi_agent(
        policies={"shared_policy"},
        policy_mapping_fn=lambda *args: "shared_policy"
    )
)

algo = config.build()

for i in range(200):
    result = algo.train()
    print(f"Iter {i}: reward={result['episode_reward_mean']}")