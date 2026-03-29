"""
Metacognitive Memory Layer — "What I Know About Myself"
Self-reflection, self-evaluation, and the agent's model of its own capabilities.
This is the layer that enables genuine self-improvement over time.
"""
import uuid
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Literal
from samsara_memory.types import (
    MemoryEntry, MemoryLayer, CapabilityProfile, SelfModel
)


class MetacognitiveMemory:
    """
    The self-model layer.

    This layer stores:
    1. What the agent believes about its own capabilities (SelfModel)
    2. How those beliefs change over time based on evidence
    3. Specific self-knowledge: "I struggle with X", "I'm reliable at Y"
    4. Meta-level lessons: what strategies work for me, what doesn't

    This is what enables:
    - "I should try a different approach for this task — my usual one failed"
    - "I have a 94% success rate at this — I'm confident, proceed"
    - "I know I tend to rush on X type of task — slow down"
    - "The last 3 times I attempted Y, I failed at step 2 — escalate early"

    Storage: PostgreSQL + SQLite (same as semantic layer, for now)
    """

    def __init__(self, vector_store, history_db, embedding_model):
        self.vector_store = vector_store
        self.db = history_db
        self.embedding = embedding_model
        self.layer = MemoryLayer.METACOGNITIVE

    # ─── Self-Model Operations ───────────────────────────────────────────────

    def get_capability_profile(self, agent_id: str, capability: str) -> Optional[Dict[str, Any]]:
        """
        Get the metacognitive profile for a specific capability.
        Returns the profile dict or None if this capability has no profile yet.
        """
        cursor = self.db.connection.cursor()
        row = cursor.execute(
            """
            SELECT capability_data FROM metacognitive_profiles
            WHERE agent_id = ? AND capability = ?
            """,
            (agent_id, capability)
        ).fetchone()
        if row:
            import json
            return json.loads(row[0])
        return None

    def update_capability_profile(
        self,
        agent_id: str,
        capability: str,
        attempt_success: bool,
        eval_passed: Optional[bool] = None,
        failure_reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Update the metacognitive profile for a capability after an attempt.

        This is called after every agent attempt, win or lose.
        It updates:
        - success_rate (exponential moving average)
        - total_attempts
        - last_success / last_attempt
        - failure_patterns (accumulated as a list)
        - eval_status: confidence | developing | unknown | declined
        """
        import json
        cursor = self.db.connection.cursor()

        existing = self.get_capability_profile(agent_id, capability)
        now = datetime.now(timezone.utc).isoformat()

        if existing:
            p = existing
            total = p.get("total_attempts", 0) + 1
            successes = p.get("successes", 0) + (1 if attempt_success else 0)
            p["total_attempts"] = total
            p["successes"] = successes
            p["success_rate"] = round(successes / total, 4)
            p["last_attempt"] = now
            if attempt_success:
                p["last_success"] = now
            if failure_reason and not attempt_success:
                patterns = p.get("failure_patterns", [])
                if failure_reason not in patterns:
                    patterns.append(failure_reason)
                p["failure_patterns"] = patterns[-10:]  # keep last 10
            p["eval_status"] = self._derive_eval_status(p)
        else:
            p = {
                "capability": capability,
                "confidence": 0.5,
                "success_rate": 1.0 if attempt_success else 0.0,
                "total_attempts": 1,
                "successes": 1 if attempt_success else 0,
                "last_attempt": now,
                "last_success": now if attempt_success else None,
                "failure_patterns": [failure_reason] if failure_reason and not attempt_success else [],
                "eval_status": "unknown",
            }

        # Store
        cursor.execute(
            """
            INSERT OR REPLACE INTO metacognitive_profiles
              (agent_id, capability, capability_data, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            (agent_id, capability, json.dumps(p), now)
        )
        self.db.connection.commit()
        return p

    def _derive_eval_status(self, profile: Dict[str, Any]) -> str:
        """Derive eval_status from profile data."""
        rate = profile.get("success_rate", 0)
        total = profile.get("total_attempts", 0)
        if total < 3:
            return "unknown"
        if rate >= 0.85:
            return "confidence"
        if rate >= 0.60:
            return "developing"
        return "declined"

    # ─── Self-Evaluation (the core Samsara Loop gate) ──────────────────────

    def evaluate_attempt(
        self,
        agent_id: str,
        capability: str,
        confidence_threshold: float = 0.85,
    ) -> Dict[str, Any]:
        """
        The Samsara Loop self-eval gate.

        Before attempting a task, the agent evaluates whether it should proceed.
        Returns:
          - can_attempt: bool — should the agent try this?
          - confidence: float — current estimated success probability
          - recommendation: str — proceed | decline | investigate
          - reason: str — human-readable explanation
          - missing_tests: bool — are there eval test cases?
        """
        profile = self.get_capability_profile(agent_id, capability)

        if not profile:
            return {
                "can_attempt": True,
                "confidence": 0.5,
                "recommendation": "proceed",
                "reason": f"No history for '{capability}' — attempt and record results to build profile",
                "missing_tests": True,
            }

        rate = profile.get("success_rate", 0)
        status = profile.get("eval_status", "unknown")
        total = profile.get("total_attempts", 0)
        patterns = profile.get("failure_patterns", [])

        if rate >= confidence_threshold and total >= 3:
            return {
                "can_attempt": True,
                "confidence": rate,
                "recommendation": "proceed",
                "reason": f"'{capability}' has {rate*100:.0f}% success rate ({total} attempts) — above {confidence_threshold*100:.0f}% threshold",
                "missing_tests": False,
            }
        elif rate < 0.60 and total >= 3:
            return {
                "can_attempt": False,
                "confidence": rate,
                "recommendation": "decline",
                "reason": f"'{capability}' has only {rate*100:.0f}% success rate — below 60% safety floor. Patterns: {', '.join(patterns[-3:]) if patterns else 'insufficient data'}",
                "missing_tests": False,
            }
        else:
            return {
                "can_attempt": True,
                "confidence": rate,
                "recommendation": "investigate",
                "reason": f"'{capability}' at {rate*100:.0f}% ({total} attempts) — below confidence threshold. Proceed with extra verification steps.",
                "missing_tests": total < 3,
            }

    # ─── Self-Knowledge Entries ────────────────────────────────────────────

    def add_self_knowledge(
        self,
        agent_id: str,
        content: str,
        category: Literal["strength", "weakness", "pattern", "lesson", "preference"],
        context: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> MemoryEntry:
        """
        Add a metacognitive self-knowledge entry.

        Examples:
        - "I tend to overcommit on tasks I'm unsure about — I should ask more questions" (weakness)
        - "When I read the full error traceback first, I debug faster" (lesson)
        - "I'm reliable at data extraction tasks but struggle with regex edge cases" (pattern)
        """
        now = datetime.now(timezone.utc).isoformat()
        entry_id = f"META-{uuid.uuid4().hex[:8]}"

        entry = MemoryEntry(
            id=entry_id,
            layer=MemoryLayer.METACOGNITIVE,
            content=content,
            agent_id=agent_id,
            created_at=now,
            tags=tags or [category],
            metadata={
                "category": category,
                "context": context,
            }
        )

        # Persist to DB
        cursor = self.db.connection.cursor()
        cursor.execute(
            """
            INSERT INTO memory_entries (id, layer, content, agent_id, created_at, tags, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (entry.id, entry.layer.value, entry.content, entry.agent_id,
             entry.created_at, ",".join(entry.tags), json.dumps(entry.metadata))
        )
        self.db.connection.commit()

        # Also index in vector store
        self.vector_store.add(
            text=entry.content,
            user_id=agent_id,
            metadata={"layer": self.layer.value, "category": category}
        )

        return entry

    def get_self_knowledge(
        self,
        agent_id: str,
        category: Optional[str] = None,
        limit: int = 20,
    ) -> List[MemoryEntry]:
        """Retrieve self-knowledge entries, optionally filtered by category."""
        cursor = self.db.connection.cursor()
        if category:
            rows = cursor.execute(
                """
                SELECT id, layer, content, agent_id, created_at, tags, metadata
                FROM memory_entries
                WHERE agent_id = ? AND layer = ? AND tags LIKE ?
                ORDER BY created_at DESC LIMIT ?
                """,
                (agent_id, self.layer.value, f"%{category}%", limit)
            ).fetchall()
        else:
            rows = cursor.execute(
                """
                SELECT id, layer, content, agent_id, created_at, tags, metadata
                FROM memory_entries
                WHERE agent_id = ? AND layer = ?
                ORDER BY created_at DESC LIMIT ?
                """,
                (agent_id, self.layer.value, limit)
            ).fetchall()

        import json
        entries = []
        for row in rows:
            entries.append(MemoryEntry(
                id=row[0], layer=MemoryLayer(row[1]), content=row[2],
                agent_id=row[3], created_at=row[4],
                tags=row[5].split(",") if row[5] else [],
                metadata=json.loads(row[6]) if row[6] else {}
            ))
        return entries

    def build_self_model(self, agent_id: str) -> SelfModel:
        """
        Construct the complete SelfModel for an agent from all metacognitive data.
        This is what Samsara Loop uses to make all self-evaluation decisions.
        """
        import json
        cursor = self.db.connection.cursor()
        rows = cursor.execute(
            "SELECT capability, capability_data FROM metacognitive_profiles WHERE agent_id = ?",
            (agent_id,)
        ).fetchall()

        capabilities = {}
        for row in rows:
            data = json.loads(row[1])
            capabilities[row[0]] = CapabilityProfile(
                agent_id=agent_id,
                capabilities={row[0]: data}
            )

        self_knowledge = self.get_self_knowledge(agent_id, limit=100)
        strengths = [e.content for e in self_knowledge if "strength" in e.tags]
        weaknesses = [e.content for e in self_knowledge if "weakness" in e.tags]

        all_rates = [c.capabilities[k].get("success_rate", 0)
                     for c in capabilities.values()
                     for k in c.capabilities]
        overall = sum(all_rates) / len(all_rates) if all_rates else 0.0

        return SelfModel(
            agent_id=agent_id,
            capabilities=capabilities,
            strengths=strengths,
            weaknesses=weaknesses,
            eval_suite_coverage=len(capabilities),
            overall_health_score=round(overall, 4),
        )
