# Agent workflows

See also `docs/diagrams/agent_workflow.mmd`.

```mermaid
flowchart LR
  P[Planner] --> R[Research]
  R --> E[Executor]
  E --> C[Critic]
  C --> F[Reflect]
  F -->|gaps| R
  F -->|ok| Out[Synthesize answer]
```

**Capabilities:** short-term memory, tool schema validation, retry on weak research, calculator recovery, human approval gate (`AIWB_REQUIRE_APPROVAL=1`).
