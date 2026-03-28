''''
Metacognitive Memory Layer — What Do I Know About Myself
'''
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
            f"Capability: {capability}
"
            f"Success rate: {success_rate:.0%} over {total_attempts} attempts
"
            f"Failure patterns: {', '.join(failure_patterns)}
"
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
