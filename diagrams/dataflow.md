## Data Flow

```mermaid
sequenceDiagram
    participant Drone_i
    participant Drone_j
    participant Env as Environment
    participant GNN1 as GNN Layer 1
    participant GNN2 as GNN Layer 2
    participant Actor
    participant Critic

    Env->>Drone_i: pos(2) + vel(2)
    Env->>Drone_j: pos(2) + vel(2)
    Drone_i->>GNN1: self (4), neighbors (n-1 × 4), mask
    GNN1->>GNN2: self_emb (64), neigh_emb (n-1 × 64)
    GNN2->>Actor: embedding (128)
    GNN2->>Critic: embedding (128)
    Actor->>Env: discrete action (0-4)
    Critic-->>Critic: value estimate
    Env->>Env: physics step, radar interaction
    Env->>Drone_i: reward + next observation
```
