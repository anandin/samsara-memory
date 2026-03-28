"""
Procedural Memory Layer — "How To Do X"
Stores skills, methods, prompt templates, how-to knowledge.
The agent uses this to answer: "How do I handle a loyalty points redemption?"
"""
import hashlib
import uuid
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from samsara_memory.types import MemoryLayer


class ProceduralMemory:
    """
    Stores how to do things — skills, methods, procedures, prompt templates.

    Procedural memory is the most stable layer. Once stored, it rarely changes.
    It's the agent's playbook for how to accomplish specific task types.

    This layer maps directly to mem0's existing procedural_memory implementation,
    but with a richer schema and agent-native query interface.
    """

    def __init__(self, vector_store, embedding_model):
        self.vector_store = vector_store
        self.embedding = embedding_model
        self.layer = MemoryLayer.PROCEDURAL

    def register_skill(
        self,
        agent_id: str,
        skill_name: str,
        procedure: str,
        trigger_conditions: Optional[List[str]] = None,
        examples: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Register a new skill or update an existing one.
        Returns the skill_id.
        """
        skill_id = str(uuid.uuid4())

        content = f"Skill: {skill_name}\nProcedure: {procedure}"
        if trigger_conditions:
            content += f"\nTrigger conditions: {', '.join(trigger_conditions)}"
        if examples:
            content += f"\nExamples: {', '.join(examples)}"

        embedding = self.embedding.embed(content, "add")

        payload = {
            "data": content,
            "layer": self.layer.value,
            "agent_id": agent_id,
            "skill_name": skill_name,
            "trigger_conditions": trigger_conditions or [],
            "examples": examples or [],
            "hash": hashlib.md5(content.encode()).hexdigest(),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        if metadata:
            payload["metadata"] = metadata

        self.vector_store.add(vectors=[embedding], payloads=[payload])
        return skill_id

    def find_skill(
        self, agent_id: str, task_description: str, limit: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Find the most relevant skill for a given task.
        The agent uses this to answer: "Do I know how to do this?"
        """
        embedding = self.embedding.embed(task_description, "search")
        filters = {"agent_id": agent_id, "layer": self.layer.value}

        results = self.vector_store.search(
            query=task_description,
            vectors=embedding,
            limit=limit,
            filters=filters,
        )
        return [
            {
                "skill_name": r.payload.get("skill_name"),
                "content": r.payload.get("data"),
                "trigger_conditions": r.payload.get("trigger_conditions", []),
                "score": r.score,
            }
            for r in results
        ]

    def get_all_skills(self, agent_id: str) -> List[str]:
        """List all registered skills for an agent."""
        all_items = self.vector_store.list(
            filters={"agent_id": agent_id, "layer": self.layer.value}, limit=500
        )
        return list(set(r.payload.get("skill_name") for r in all_items if r.payload.get("skill_name")))

    def skill_exists(self, agent_id: str, skill_name: str) -> bool:
        """Check if a specific skill has been registered."""
        items = self.vector_store.list(
            filters={
                "agent_id": agent_id,
                "layer": self.layer.value,
                "skill_name": skill_name,
            },
            limit=1,
        )
        return len(items) > 0
