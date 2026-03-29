"""
SamsaraMemory — The 5-Layer Cognitive Memory System

The main API. This is what agents interact with.
Instantiate with: SamsaraMemory(agent_id="my-agent")

Layers (outer to inner):
    EPISODIC      → "What happened" — execution traces, failure stories
    SEMANTIC      → "What is true" — facts, entities, knowledge
    PROCEDURAL    → "How to do X" — skills, methods, patterns
    WORKING       → "What I'm thinking now" — scratch pad (ephemeral)
    METACOGNITIVE → "What I know about myself" — self-model, self-eval

Samsara Loop integration:
    Before attempting a capability → self_evaluate(agent_id, capability)
    After success/failure        → record_attempt(agent_id, capability, success)
    When human corrects you       → capture_correction(agent_id, ...)
    When you hit a knowledge gap   → capture_knowledge_gap(agent_id, ...)
"""
from typing import List, Optional, Dict, Any, Literal
from samsara_memory.types import (
    MemoryEntry, TrajectoryEvent, ToolCall, SelfModel, CapabilityProfile
)


class SamsaraMemory:
    """
    5-layer memory system for persistent agent intelligence.

    Agents interact with this single class. All 5 layers are
    instantiated together and share the same underlying DB + vector store.
    """

    def __init__(
        self,
        agent_id: str,
        db_path: str = "./samsara_memory.db",
        vector_store = None,  # pass None to use default in-memory
        embedding_model = None,
    ):
        self.agent_id = agent_id

        # Import here so the package works even if dependencies aren't installed
        from samsara_memory.layers.episodic import EpisodicMemory
        from samsara_memory.layers.semantic import SemanticMemory
        from samsara_memory.layers.procedural import ProceduralMemory
        from samsara_memory.layers.working import WorkingMemory
        from samsara_memory.layers.metacognitive import MetacognitiveMemory
        from samsara_memory.db.database import Database

        # Shared DB instance
        self.db = Database(db_path)
        self.db.init_schema()

        # Default embedding (OpenAI-compatible)
        if embedding_model is None:
            try:
                from samsara_memory.embedding import DefaultEmbedding
                embedding_model = DefaultEmbedding()
            except ImportError:
                embedding_model = _DummyEmbedding()

        # Default vector store
        if vector_store is None:
            try:
                from samsara_memory.vectorstore import InMemoryVectorStore
                vector_store = InMemoryVectorStore()
            except ImportError:
                vector_store = _DummyVectorStore()

        # 5 layers — all share self.db and self.vector_store
        self.episodic = EpisodicMemory(vector_store, self.db, embedding_model)
        self.semantic = SemanticMemory(vector_store, self.db, embedding_model)
        self.procedural = ProceduralMemory(vector_store, self.db, embedding_model)
        self.working = WorkingMemory()  # ephemeral — no DB needed
        self.metacognitive = MetacognitiveMemory(vector_store, self.db, embedding_model)

    # ─── Samsara Loop: Self-Evaluation Gate ─────────────────────────────────

    def self_evaluate(
        self,
        capability: str,
        confidence_threshold: float = 0.85,
    ) -> Dict[str, Any]:
        """
        The Samsara Loop gate — call BEFORE attempting a task.

        Returns a dict with:
          - can_attempt: bool
          - confidence: float (0-1)
          - recommendation: str (proceed | decline | investigate)
          - reason: str
          - missing_tests: bool
        """
        return self.metacognitive.evaluate_attempt(
            self.agent_id, capability, confidence_threshold
        )

    def record_attempt(
        self,
        capability: str,
        success: bool,
        failure_reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Call AFTER an attempt to update the self-model.
        Records the outcome and updates the capability profile.
        """
        return self.metacognitive.update_capability_profile(
            self.agent_id, capability, attempt_success=success,
            failure_reason=failure_reason
        )

    # ─── Samsara Loop: Capture Operations ──────────────────────────────────

    def capture_error(
        self,
        trajectory: List[ToolCall],
        error_message: str,
        task: str,
        capability: Optional[str] = None,
    ) -> str:
        """
        Capture a failure into episodic + trigger metacognitive update.
        This is the Samsara Loop "learn from failure" entry point.
        """
        import uuid
        from datetime import datetime, timezone

        # Store in episodic
        outcome = "failure"
        failure_step = next((i for i, s in enumerate(reversed(trajectory)) if not s.success), None)

        event = TrajectoryEvent(
            session_id=str(uuid.uuid4()),
            agent_id=self.agent_id,
            task=task,
            steps=trajectory,
            outcome=outcome,
            failure_step=failure_step,
            failure_reason=error_message,
            duration_ms=sum(s.latency_ms for s in trajectory),
        )
        eid = self.episodic.store_trajectory(event)

        # Update metacognitive profile
        if capability:
            self.metacognitive.update_capability_profile(
                self.agent_id, capability, attempt_success=False,
                failure_reason=error_message
            )
        return eid

    def capture_success(
        self,
        capability: str,
        trajectory: List[ToolCall],
        task: str,
        confidence_delta: Optional[float] = None,
    ) -> str:
        """
        Capture a success into episodic + update metacognitive profile.
        """
        import uuid
        event = TrajectoryEvent(
            session_id=str(uuid.uuid4()),
            agent_id=self.agent_id,
            task=task,
            steps=trajectory,
            outcome="success",
            duration_ms=sum(s.latency_ms for s in trajectory),
        )
        eid = self.episodic.store_trajectory(event)
        self.metacognitive.update_capability_profile(
            self.agent_id, capability, attempt_success=True
        )
        return eid

    def capture_correction(
        self,
        human_feedback: str,
        original_action: str,
        context: Optional[str] = None,
    ) -> str:
        """
        Capture a human correction — the strongest learning signal.
        Stored in semantic layer as a verified fact.
        """
        content = (
            f"[CORRECTION] Human said: '{human_feedback}'. "
            f"Original action was: '{original_action}'. "
            f"Context: {context or 'N/A'}"
        )
        return self.semantic.add(
            content=content,
            user_id=self.agent_id,
            category="human_correction",
            tags=["correction", "human_signal", "verified"],
        )

    def capture_best_practice(
        self,
        content: str,
        capability: Optional[str] = None,
        source: str = "agent_experience",
    ) -> str:
        """
        Promote a discovered pattern to best-practice status.
        """
        tags = ["best_practice", source]
        if capability:
            tags.append(capability)
        return self.semantic.add(
            content=content,
            user_id=self.agent_id,
            category="best_practice",
            tags=tags,
        )

    def capture_knowledge_gap(
        self,
        question: str,
        context: Optional[str] = None,
        priority: int = 1,
    ) -> str:
        """
        Log a knowledge gap for later resolution.
        """
        content = f"[KNOWLEDGE GAP] Q: {question}"
        if context:
            content += f" | Context: {context}"
        return self.semantic.add(
            content=content,
            user_id=self.agent_id,
            category="knowledge_gap",
            tags=["gap", f"priority_{priority}"],
        )

    # ─── Retrieval ──────────────────────────────────────────────────────────

    def recall(
        self,
        query: str,
        layers: Optional[List[str]] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Search across all memory layers.

        layers: list of "episodic" | "semantic" | "procedural" | "working" | "metacognitive"
                None = search all layers
        """
        layers = layers or ["episodic", "semantic", "procedural", "metacognitive"]
        results = []

        if "episodic" in layers:
            results.extend(self.episodic.search(query, limit=limit))

        if "semantic" in layers:
            results.extend(self.semantic.search(query, user_id=self.agent_id, limit=limit))

        if "procedural" in layers:
            results.extend(self.procedural.search(query, limit=limit))

        if "metacognitive" in layers:
            entries = self.metacognitive.get_self_knowledge(self.agent_id, limit=limit)
            results.extend([
                {"id": e.id, "layer": e.layer.value, "content": e.content, "score": 1.0, "metadata": e.metadata}
                for e in entries if query.lower() in e.content.lower()
            ])

        if "working" in layers:
            for slot, data in self.working.read_all().items():
                if query.lower() in data.lower():
                    results.append({"id": f"WM-{slot}", "layer": "working", "content": data, "score": 0.9, "metadata": {"slot": slot}})

        # Sort by score descending, deduplicate by content
        seen = set()
        deduped = []
        for r in sorted(results, key=lambda x: x.get("score", 0), reverse=True):
            if r["content"][:80] not in seen:
                seen.add(r["content"][:80])
                deduped.append(r)
        return deduped[:limit]

    def get_profile(self) -> Dict[str, Any]:
        """
        Get the full agent profile across all layers.
        Used by the web dashboard.
        """
        return {
            "agent_id": self.agent_id,
            "self_model": self.metacognitive.build_self_model(self.agent_id).model_dump(),
            "episodic_summary": self.episodic.get_recent_events(limit=10),
            "best_practices": self.semantic.get_best_practices(limit=10),
            "pending_tests": self._get_pending_tests(),
            "working_memory": self.working.summary(),
        }

    def _get_pending_tests(self) -> List[Dict[str, Any]]:
        """Get pending test cases from the database."""
        cursor = self.db.connection.cursor()
        rows = cursor.execute(
            "SELECT id, capability, test_case, created_at FROM pending_tests ORDER BY created_at DESC LIMIT 20"
        ).fetchall()
        return [
            {"id": r[0], "capability": r[1], "test_case": r[2], "created_at": r[3]}
            for r in rows
        ]

    # ─── Working Memory Shortcuts ──────────────────────────────────────────

    def think(self, slot: str, content: str) -> str:
        """Short-hand for working memory write."""
        return self.working.write(slot, content)

    def think_of(self, slot: str) -> Optional[str]:
        """Short-hand for working memory read."""
        return self.working.read(slot)

    def finish_thought(self, slot: str, promote: bool = True) -> Optional[Dict[str, Any]]:
        """
        Finish a working memory slot — optionally promote to semantic memory.
        """
        if promote:
            return self.working.promote_to_episodic_format(slot, self.agent_id)
        else:
            self.working.delete(slot)
            return None

    # ─── Utility ────────────────────────────────────────────────────────────

    def health_check(self) -> Dict[str, Any]:
        """Return system health status for all layers."""
        return {
            "agent_id": self.agent_id,
            "episodic": "ok" if self.episodic else "error",
            "semantic": "ok" if self.semantic else "error",
            "procedural": "ok" if self.procedural else "error",
            "working": self.working.summary(),
            "metacognitive": "ok",
        }

    def clear_all(self) -> None:
        """Clear ALL layers — use with caution."""
        self.episodic.clear()
        self.semantic.clear()
        self.procedural.clear()
        self.working.clear()
        cursor = self.db.connection.cursor()
        cursor.execute("DELETE FROM memory_entries WHERE agent_id = ?", (self.agent_id,))
        self.db.connection.commit()


# ─── Dummy fallbacks for optional dependencies ────────────────────────────

class _DummyEmbedding:
    def encode(self, text: str) -> List[float]:
        import hashlib
        h = hashlib.sha256(text.encode()).digest()
        return [b / 255.0 for b in h[:32]]  # 32-dim fake embedding

class _DummyVectorStore:
    def add(self, text: str, user_id: str, metadata: dict = None):
        pass
    def search(self, query: str, user_id: str, limit: int = 5):
        return []
