
---

# 📊 diagrams/dataflow.md

```markdown
## Data Flow

```mermaid
sequenceDiagram
    participant Drone1
    participant Drone2
    participant GNN
    participant Policy

    Drone1->>GNN: state + neighbors
    Drone2->>GNN: state + neighbors
    GNN->>Policy: embeddings
    Policy->>Drone1: action
    Policy->>Drone2: action