"""
Semantic Memory Layer — "What Is True"
Stores facts, entities, relationships about the world.
Used for: "What is the difference between disputed and refunded?"
"""
import hashlib
import uuid
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from samsara_memory.types import MemoryLayer


class SemanticMemory:
    """
    Stores what is true — facts, entities, relationships.

    Semantic memory is the agent's knowledge base about the world.
    Unlike episodic (what happened), semantic stores what is generally true.

    Interface compatible with mem0's main Memory.add() for easy migration.
    """

    def __init__(self, vector_store, embedding_model, llm):
        self.vector_store = vector_store
        self.embedding = embedding_model
        self.llm = llm
        self.layer = MemoryLayer.SEMANTIC

    def add(
        self,
        agent_id: str,
        facts: List[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        Add facts to semantic memory.
        facts: list of factual statements to store.
        Returns list of memory entry IDs.
        """
        ids = []
        for fact in facts:
            fact_id = str(uuid.uuid4())
            embedding = self.embedding.embed(fact, "add")

            payload = {
                "data": fact,
                "layer": self.layer.value,
                "agent_id": agent_id,
                "hash": hashlib.md5(fact.encode()).hexdigest(),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            if metadata:
                payload["metadata"] = metadata

            self.vector_store.add(vectors=[embedding], payloads=[payload])
            ids.append(fact_id)
        return ids

    def search(
        self, agent_id: str, query: str, limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search semantic memory for relevant facts.
        The agent uses this to answer: "What do I know about X?"
        """
        embedding = self.embedding.embed(query, "search")
        filters = {"agent_id": agent_id, "layer": self.layer.value}

        results = self.vector_store.search(
            query=query, vectors=embedding, limit=limit, filters=filters
        )
        return [{"fact": r.payload.get("data"), "id": r.id, "score": r.score} for r in results]

    def get_all(self, agent_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all semantic memories for an agent."""
        all_memories = self.vector_store.list(
            filters={"agent_id": agent_id, "layer": self.layer.value}, limit=limit
        )
        return [{"fact": m.payload.get("data"), "id": m.id} for m in all_memories]
