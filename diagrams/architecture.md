## System Architecture

```mermaid
graph TD
    A[Drone Observations] --> B[Graph Construction]
    B --> C[GNN Layer 1]
    C --> D[GNN Layer 2]
    D --> E[Agent Embeddings]
    E --> F[Actor Network]
    E --> G[Centralized Critic]
    F --> H[Actions]
    G --> I[Value Function]