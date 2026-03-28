# Samsara Memory

**5-layer cognitive memory for AI agents.**

Forked from [mem0](https://github.com/mem0ai/mem0) (Apache 2.0) with major extensions.

## The 5 Layers

| Layer | Question | Example |
|-------|----------|---------|
| **Episodic** | What happened? | "Failed on Step 3 — wrong tool" |
| **Semantic** | What is true? | "Disputed charges escalate to supervisor" |
| **Procedural** | How do I do X? | "Process a loyalty redemption" |
| **Working** | What am I thinking now? | Current task, active goals |
| **Metacognitive** | Am I good at this? | "94% on refunds, 61% on cross-sell" |

## Quick Start

```python
from samsara_memory import SamsaraMemory

memory = SamsaraMemory(agent_id="support-agent-1")

# Before attempting
report = memory.prepare_for_task("customer asking about disputed charge")
print(report["recommendation"])

# After completion
memory.store_trajectory(trajectory_event)
memory.update_capability("disputed_charge", 0.87, 52, ["wrong escalation"])

# Ask yourself
should_act, confidence, reason = memory.should_attempt("process refund")
```

## Agent-Native API

The agent calls these directly:
- `prepare_for_task(task)` — readiness report before attempting
- `should_attempt(task)` — (yes/no, confidence, reason)
- `store_trajectory(event)` — save a failure or success
- `recall_similar(task)` — have I done this before?
- `find_skill(task)` — do I know how?
- `update_capability(...)` — track my own improvement

## Architecture

```
Agent (any framework)
    via MCP
Samsara Memory
    Episodic + Semantic + Procedural + Working + Metacognitive
    Vector Store + SQLite
```

Status: Pre-alpha. Core layers implemented.
License: Apache 2.0
