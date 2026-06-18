# Drone Swarm GNN-RL

Multi-agent reinforcement learning for drone swarms using a Graph Neural Network (GNN) with PPO in Ray RLlib. Drones cooperatively neutralize moving radar targets in a 2D environment using continuous acceleration control. The architecture uses a GNN with attention and edge features (relative position, distance) for communication, a centralized critic that observes the full global state for stable value estimation, and a count-based intrinsic exploration bonus to encourage coverage. Radars are dynamic — they wander and evade nearby drones — making the task more challenging and realistic.

## Architecture

```
Drone Observations → Graph Construction → GNN Layer 1 (with edge features) → GNN Layer 2
→ Agent Embeddings → Actor Network → Continuous Actions
                   → Centralized Critic (global state) → Value Function
```

Each drone observes its own position/velocity and communicates with nearby drones (within `comm_radius`). A GNN with attention and edge features (relative position, distance) aggregates neighbor information. The actor produces continuous acceleration commands, while a centralized critic sees the full global state for value estimation.

## Project Structure

```
├── env/drone_env.py          # Multi-agent environment (DroneSwarmEnv)
├── models/
│   ├── gnn_layer.py          # GNN layer with attention, edge features, LayerNorm
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

The environment resets when all radars are destroyed or `max_steps` is reached.

## Environment

- **Action space**: `Box(-1, 1, shape=(2,))` — continuous acceleration (ax, ay), scaled by `max_acceleration`
- **Observation**: Dict with:
  - `self` (4) — own position + velocity
  - `neighbors` (n-1 × 4) — nearby drones' pos + vel (zero-padded with mask)
  - `mask` (n-1) — visibility mask for neighbors
  - `global_state` (n × 4) — all agents' pos + vel concatenated (for centralized critic)
- **Termination**: All radars destroyed or `max_steps` exceeded

## Radars

Radars are dynamic: they wander randomly and actively evade nearby drones within a detect range. Drones destroy a radar by getting close (dist < 5) with low velocity (jamming).

## Rewards

| Source | Value | Description |
|---|---|---|
| Step bonus | +1 | Per agent per step |
| Proximity | (20 - dist) / 20 × 5 | Closer to radar = more reward (dist < 20) |
| Destroy | +20 | Radar destroyed |
| Intrinsic | 1 / sqrt(count+1) × 0.01 | Count-based exploration bonus (per grid cell) |
