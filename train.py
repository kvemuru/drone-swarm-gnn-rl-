import os
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

num_workers = min(cfg["training"]["num_workers"], os.cpu_count() or 1)

config = (
    PPOConfig()
    .api_stack(
        enable_rl_module_and_learner=False,
        enable_env_runner_and_connector_v2=False,
    )
    .environment(env=DroneSwarmEnv, env_config=cfg)
    .framework("torch")
    .env_runners(num_env_runners=num_workers)
    .training(
        model={
            "custom_model": "gnn_mappo",
            "custom_model_config": {
                "comm_radius": cfg["communication"]["comm_radius"],
                "n_agents": cfg["env"]["num_drones"],
            },
        },
        train_batch_size=cfg["training"]["train_batch_size"],
        gamma=cfg["training"]["gamma"],
        lr=cfg["training"]["lr"],
    )
    .multi_agent(
        policies={"shared_policy"},
        policy_mapping_fn=lambda *args, **kwargs: "shared_policy",
    )
)

algo = config.build()

for i in range(200):
    result = algo.train()
    reward = result.get("env_runners", {}).get("episode_reward_mean")
    print(f"Iter {i}: reward={reward}")
