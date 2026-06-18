# Drone Swarm GNN-RL

Multi-agent reinforcement learning for drone swarms using Graph Neural Networks (GNN) with PPO. Drones cooperatively neutralize radar targets in a 2D environment.

## Architecture

```
Drone Observations → Graph Construction → GNN Layer 1 → GNN Layer 2
→ Agent Embeddings → Actor Network → Actions
                   → Centralized Critic → Value Function
```

Each drone observes its own position/velocity and communicates with nearby drones (within `comm_radius`). A GNN with attention aggregates neighbor information to produce shared embeddings, which feed into a shared PPO policy with a centralized value function.

## Project Structure

```
├── env/drone_env.py          # Multi-agent environment (DroneSwarmEnv)
├── models/
│   ├── gnn_layer.py          # GNN layer with attention
│   └── gnn_mappo_model.py    # Custom TorchModelV2 for RLlib
├── config.yaml               # Hyperparameters
├── train.py                  # Training entry point
├── requirements.txt
└── diagrams/                 # Architecture & data flow diagrams
```

## Setup

```bash
pip install "ray[rllib]" torch numpy gymnasium pyyaml
```

## Configuration

Edit `config.yaml`:

| Parameter | Description | Default |
|---|---|---|
| `num_drones` | Number of agents | 4 |
| `num_radars` | Number of targets | 3 |
| `map_size` | World size | 200 |
| `max_steps` | Episode length | 800 |
| `comm_radius` | Communication range | 30 |
| `train_batch_size` | PPO batch size | 8000 |
| `lr` | Learning rate | 3e-4 |
| `gamma` | Discount factor | 0.99 |

## Usage

```bash
python train.py
```

The environment resets when all radars are destroyed or `max_steps` is reached. Drones receive reward for proximity to radars (+10) and destroying them (+20), plus a step bonus (+1).

## Environment

- **Action space**: `Discrete(5)` — move N/S/E/W or stay
- **Observation**: Dict with `self` (pos+vel, 4-dim), `neighbors` (n-1 agents × 4), and `mask` (visibility)
- **Termination**: All radars destroyed or max steps exceeded
