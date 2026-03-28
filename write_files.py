import os

files = {}

files["/workspace/samsara-memory/samsara_memory/layers/working.py"] = """\
'\'\'\'
Working Memory Layer — What Am I Thinking About Right Now
\'\'\'
import uuid
from datetime import datetime, timezone
from samsara_memory.types import MemoryLayer


class WorkingMemory:
    def __init__(self, vector_store, embedding_model):
        self.vector_store = vector_store
        self.embedding = embedding_model
        self.layer = MemoryLayer.WORKING

    def push(self, agent_id, session_id, content, focus=None, goal=None):
        entry_id = f"work_{uuid.uuid4().hex[:12]}"
        embedding = self.embedding.embed(content, "add")
        payload = {
            "data": content, "layer": self.layer.value,
            "agent_id": agent_id, "session_id": session_id,
            "focus": focus or "", "goal": goal or "",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "is_active": True,
        }
        self.vector_store.add(vectors=[embedding], payloads=[payload])
        return entry_id

    def get_active(self, agent_id, session_id):
        items = self.vector_store.list(
            filters={"agent_id": agent_id, "session_id": session_id, "layer": self.layer.value}, limit=10)
        if not items:
            return {"focus": None, "goals": [], "context": []}
        return {
            "focus": items[0].payload.get("focus"),
            "goals": [i.payload.get("goal") for i in items if i.payload.get("goal")],
            "context": [i.payload.get("data") for i in items],
        }

    def clear_session(self, agent_id, session_id):
        items = self.vector_store.list(
            filters={"agent_id": agent_id, "session_id": session_id, "layer": self.layer.value}, limit=500)
        count = 0
        for item in items:
            self.vector_store.delete(vector_id=item.id)
            count += 1
        return count
"""

files["/workspace/samsara-memory/samsara_memory/layers/metacognitive.py"] = """\
'\'\''
Metacognitive Memory Layer — What Do I Know About Myself
\'\'\'
import hashlib
import uuid
from datetime import datetime, timezone
from samsara_memory.types import MemoryLayer


class MetacognitiveMemory:
    def __init__(self, vector_store, embedding_model):
        self.vector_store = vector_store
        self.embedding = embedding_model
        self.layer = MemoryLayer.METACOGNITIVE

    def update_capability(self, agent_id, capability, success_rate, total_attempts,
                          failure_patterns, eval_status="developing"):
        entry_id = str(uuid.uuid4())
        content = (
            f"Capability: {capability}\n"
            f"Success rate: {success_rate:.0%} over {total_attempts} attempts\n"
            f"Failure patterns: {', '.join(failure_patterns)}\n"
            f"Eval status: {eval_status}"
        )
        embedding = self.embedding.embed(content, "add")
        payload = {
            "data": content, "layer": self.layer.value, "agent_id": agent_id,
            "capability": capability, "success_rate": success_rate,
            "total_attempts": total_attempts,
            "failure_patterns": failure_patterns, "eval_status": eval_status,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self.vector_store.add(vectors=[embedding], payloads=[payload])
        return entry_id

    def should_attempt(self, agent_id, task):
        embedding = self.embedding.embed(task, "search")
        results = self.vector_store.search(
            query=task, vectors=embedding, limit=3,
            filters={"agent_id": agent_id, "layer": self.layer.value})
        if not results:
            return True, 0.5, "No prior data"
        top = results[0]
        rate = top.payload.get("success_rate", 0.5)
        cap = top.payload.get("capability", "unknown")
        if rate >= 0.85:
            return True, rate, f"Strong track record on {cap}"
        elif rate >= 0.6:
            return True, rate, f"Moderate track record"
        elif rate >= 0.3:
            return False, rate, f"Low success rate on {cap}"
        else:
            return False, rate, f"High failure rate on {cap}"

    def get_self_model(self, agent_id):
        items = self.vector_store.list(
            filters={"agent_id": agent_id, "layer": self.layer.value}, limit=200)
        caps = {}
        for item in items:
            cap = item.payload.get("capability")
            if not cap or cap in caps:
                continue
            caps[cap] = {
                "success_rate": item.payload.get("success_rate", 0),
                "total_attempts": item.payload.get("total_attempts", 0),
                "failure_patterns": item.payload.get("failure_patterns", []),
                "eval_status": item.payload.get("eval_status", "unknown"),
            }
        return {
            "agent_id": agent_id, "capabilities": caps,
            "strengths": [c for c, v in caps.items() if v["eval_status"] == "confident"],
            "developing": [c for c, v in caps.items() if v["eval_status"] == "developing"],
            "unknown": [c for c, v in caps.items() if v["eval_status"] == "unknown"],
            "total_capabilities_tracked": len(caps),
        }
"""

files["/workspace/samsara-memory/samsara_memory/core.py"] = """\
import uuid
from typing import Dict, List, Any
from samsara_memory.layers.episodic import EpisodicMemory
from samsara_memory.layers.semantic import SemanticMemory
from samsara_memory.layers.procedural import ProceduralMemory
from samsara_memory.layers.working import WorkingMemory
from samsara_memory.layers.metacognitive import MetacognitiveMemory


class SamsaraMemory:
    def __init__(self, agent_id, vector_store=None, embedding_model=None, llm=None, history_db=None):
        self.agent_id = agent_id
        self.session_id = str(uuid.uuid4())
        self._vector_store = vector_store
        self._embedding = embedding_model
        self._llm = llm
        self._history_db = history_db
        if vector_store and embedding_model and llm:
            self._init_layers()

    def _init_layers(self):
        self.episodic = EpisodicMemory(self._vector_store, self._history_db, self._embedding)
        self.semantic = SemanticMemory(self._vector_store, self._embedding, self._llm)
        self.procedural = ProceduralMemory(self._vector_store, self._embedding)
        self.working = WorkingMemory(self._vector_store, self._embedding)
        self.metacognitive = MetacognitiveMemory(self._vector_store, self._embedding)

    def store_trajectory(self, event):
        return self.episodic.store_trajectory(event)

    def recall_similar(self, task, outcome_filter=None):
        return self.episodic.query(self.agent_id, task, outcome_filter=outcome_filter)

    def learn_fact(self, facts, metadata=None):
        return self.semantic.add(self.agent_id, facts, metadata)

    def query_knowledge(self, query):
        return self.semantic.search(self.agent_id, query)

    def register_skill(self, skill_name, procedure, trigger_conditions=None):
        return self.procedural.register_skill(self.agent_id, skill_name, procedure, trigger_conditions)

    def find_skill(self, task):
        return self.procedural.find_skill(self.agent_id, task)

    def push_working(self, content, focus=None, goal=None):
        return self.working.push(self.agent_id, self.session_id, content, focus, goal)

    def get_working_state(self):
        return self.working.get_active(self.agent_id, self.session_id)

    def update_capability(self, capability, success_rate, total_attempts,
                          failure_patterns, eval_status="developing"):
        return self.metacognitive.update_capability(
            self.agent_id, capability, success_rate, total_attempts,
            failure_patterns, eval_status)

    def should_attempt(self, task):
        return self.metacognitive.should_attempt(self.agent_id, task)

    def get_self_model(self):
        return self.metacognitive.get_self_model(self.agent_id)

    def prepare_for_task(self, task):
        should_act, confidence, reason = self.should_attempt(task)
        similar = self.recall_similar(task)
        skills = self.find_skill(task)
        knowledge = self.query_knowledge(task)
        failures = [s for s in similar if s.get("outcome") == "failure"][:3]
        if not should_act:
            rec = "Decline or escalate"
        elif failures:
            rec = f"Caution: {len(failures)} similar failures on record"
        elif skills:
            rec = "Proceed: relevant skills found"
        else:
            rec = "Proceed: no specific risk factors"
        return {
            "task": task, "should_attempt": should_act,
            "self_confidence": confidence, "reason": reason,
            "relevant_skills": skills, "knowledge_base": knowledge,
            "similar_past_experiences": similar, "recent_failures": failures,
            "recommendation": rec,
        }

    def session_summary(self):
        return {
            "agent_id": self.agent_id, "session_id": self.session_id,
            "working_state": self.get_working_state(),
            "self_model": self.get_self_model(),
        }
"""

files["/workspace/samsara-memory/pyproject.toml"] = """\
[project]
name = "samsara-memory"
version = "0.1.0"
description = "5-layer cognitive memory for AI agents"
license = {text = "Apache 2.0"}
requires-python = ">=3.10"
dependencies = ["pydantic>=2.0"]

[project.optional-dependencies]
mem0 = ["mem0ai"]

[build-system]
requires = ["setuptools>=45", "wheel"]
build-backend = "setuptools.build_meta"
"""

files["/workspace/samsara-memory/README.md"] = """\
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
"""

files["/workspace/samsara-memory/COMPETITIVE.md"] = """\
# Samsara Memory vs Alternatives

| Feature | mem0 | Samsara Memory |
|---------|------|---------------|
| Layers | 3 | **5** |
| Agent queries own memory | No | **Yes** |
| Failure trajectory storage | No | **Yes** |
| Capability self-model | No | **Yes** |
| should_attempt() | No | **Yes** |
| prepare_for_task() | No | **Yes** |
| Metacognitive layer | No | **Yes** |

Forked from: mem0 51k stars (Apache 2.0).
We add: 2 new layers + agent-native API.
"""

for filepath, content in files.items():
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        f.write(content)
    print(f"WROTE: {filepath}")

print("\\nAll files written successfully.")
