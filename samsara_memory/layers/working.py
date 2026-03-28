''''
Working Memory Layer — What Am I Thinking About Right Now
'''
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
