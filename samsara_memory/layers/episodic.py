"""
Episodic Memory Layer — "What Happened"
Stores step-by-step event sequences: tool calls, decisions, outcomes.
The most important layer for failure analysis and test generation.
"""
import uuid
import hashlib
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from samsara_memory.types import TrajectoryEvent, ToolCall, MemoryEntry, MemoryLayer


class EpisodicMemory:
    """
    Stores what happened, step by step.

    Unlike chat history (which stores messages), episodic memory stores
    the actual execution trace: which tools were called, in what order,
    what the inputs were, what the outputs were, and whether it worked.

    This is the layer that lets an agent answer:
    - "Have I handled this type of task before?"
    - "What went wrong the last time I tried this?"
    - "Which step specifically failed?"

    Storage: PostgreSQL via mem0's vector store + SQLite history DB
    """

    def __init__(self, vector_store, history_db, embedding_model):
        self.vector_store = vector_store
        self.db = history_db
        self.embedding = embedding_model
        self.layer = MemoryLayer.EPISODIC

    def store_trajectory(self, event: TrajectoryEvent) -> str:
        """
        Store a complete execution episode in episodic memory.
        Returns the trajectory_id.
        """
        trajectory_id = str(uuid.uuid4())

        # Step-by-step narrative for semantic retrieval
        step_summaries = []
        for step in event.steps:
            status = "✓" if step.success else "✗"
            summary = f"[Step {step.step}] {status} {step.tool_name}"
            if step.error:
                summary += f" — ERROR: {step.error}"
            step_summaries.append(summary)

        # Compose the episodic memory entry
        narrative = (
            f"Task: {event.task}\n"
            f"Outcome: {event.outcome}\n"
            + ("\n".join(step_summaries))
        )

        if event.failure_reason:
            narrative += f"\nFailure reason: {event.failure_reason}"

        # Embed and store in vector store
        embedding = self.embedding.embed(narrative, "add")

        payload = {
            "data": narrative,
            "layer": self.layer.value,
            "agent_id": event.agent_id,
            "session_id": event.session_id,
            "task": event.task,
            "outcome": event.outcome,
            "failure_step": event.failure_step,
            "failure_reason": event.failure_reason,
            "duration_ms": event.duration_ms,
            "confidence_before": event.confidence_before,
            "confidence_after": event.confidence_after,
            "metadata": event.metadata,
            "hash": hashlib.md5(narrative.encode()).hexdigest(),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        self.vector_store.add(vectors=[embedding], payloads=[payload])
        return trajectory_id

    def query(
        self,
        agent_id: str,
        query: str,
        limit: int = 5,
        outcome_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query episodic memory for similar past experiences.
        The agent uses this to answer: "Have I done this before?"
        """
        embedding = self.embedding.embed(query, "search")

        filters = {"agent_id": agent_id, "layer": self.layer.value}
        if outcome_filter:
            filters["outcome"] = outcome_filter

        results = self.vector_store.search(
            query=query, vectors=embedding, limit=limit, filters=filters
        )

        return [
            {
                "task": r.payload.get("task"),
                "outcome": r.payload.get("outcome"),
                "failure_reason": r.payload.get("failure_reason"),
                "session_id": r.payload.get("session_id"),
                "similarity": r.score,
            }
            for r in results
        ]

    def get_recent_failures(self, agent_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get the most recent failures for this agent."""
        return self.query(agent_id, "failure error bug failed", limit=limit, outcome_filter="failure")

    def count_by_outcome(self, agent_id: str) -> Dict[str, int]:
        """Count episodes by outcome type — used for capability profiling."""
        # This would be a SQL aggregation in production
        # Placeholder for the interface
        return {
            "success": 0,
            "failure": 0,
            "partial": 0,
        }
