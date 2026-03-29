"""
Working Memory Layer — "What I'm Thinking Now"
A lightweight scratch pad for the current session.
Think of it as the agent's notepad during a task.
"""
import uuid
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from samsara_memory.types import MemoryEntry, MemoryLayer


class WorkingMemory:
    """
    Ephemeral, high-speed scratch pad for active thinking.

    Think of it as the whiteboard you wipe clean between tasks —
    except parts of it (conclusions, decisions, unfinished thoughts)
    can be promoted to episodic or semantic memory for long-term retention.

    This layer is what enables the agent to:
    - Hold a multi-step plan across tool calls without losing the thread
    - Remember partial results from earlier steps in a complex task
    - Track "I've decided to do X but haven't verified Y yet"
    - Keep track of what the human emphasized ("they care about X")

    Storage: In-process dict (not persisted to DB — that's the point.
    For session-persistent working memory, use episodic promotion).
    """

    def __init__(self):
        self._slots: Dict[str, Dict[str, Any]] = {}
        # "slot" = a named memory cell with a TTL
        # _slots[slot_id] = {content, created_at, tags, access_count}

    def write(self, slot: str, content: str, tags: Optional[List[str]] = None) -> str:
        """
        Write to a named working memory slot.
        Slots are overwritten if they already exist (update语义).
        """
        now = datetime.now(timezone.utc).isoformat()
        entry_id = f"WM-{slot}"

        self._slots[slot] = {
            "content": content,
            "created_at": now,
            "updated_at": now,
            "tags": tags or [],
            "access_count": 0,
        }
        return entry_id

    def read(self, slot: str) -> Optional[str]:
        """Read the content of a named slot. Returns None if empty/undefined."""
        if slot in self._slots:
            self._slots[slot]["access_count"] += 1
            return self._slots[slot]["content"]
        return None

    def read_all(self) -> Dict[str, str]:
        """Read all slots as a dict of slot_name -> content."""
        return {slot: data["content"] for slot, data in self._slots.items()}

    def delete(self, slot: str) -> bool:
        """Delete a named slot. Returns True if it existed."""
        return self._slots.pop(slot, None) is not None

    def clear(self) -> int:
        """
        Clear all slots. Returns the number of slots cleared.
        Call this at the start of a new task if you want a clean slate,
        or selectively delete individual slots after promoting their
        contents to episodic/semantic memory.
        """
        count = len(self._slots)
        self._slots.clear()
        return count

    def entries(self) -> List[MemoryEntry]:
        """Export all current slots as MemoryEntry objects."""
        now = datetime.now(timezone.utc).isoformat()
        entries = []
        for slot, data in self._slots.items():
            entries.append(MemoryEntry(
                id=f"WM-{slot}",
                layer=MemoryLayer.WORKING,
                content=data["content"],
                agent_id="local",  # working memory is session-local
                created_at=data["created_at"],
                updated_at=data["updated_at"],
                tags=data["tags"],
                metadata={
                    "slot": slot,
                    "access_count": data["access_count"],
                }
            ))
        return entries

    # ─── Promotion ───────────────────────────────────────────────────────────

    def promote_to_episodic_format(self, slot: str, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        Promote a working memory slot to episodic memory format.
        Returns a dict ready to be stored as a TrajectoryEvent or MemoryEntry.
        Returns None if the slot doesn't exist.
        """
        if slot not in self._slots:
            return None
        data = self._slots[slot]
        return {
            "id": f"WM-promoted-{slot}-{uuid.uuid4().hex[:8]}",
            "layer": MemoryLayer.WORKING,
            "content": data["content"],
            "agent_id": agent_id,
            "created_at": data["created_at"],
            "tags": ["promoted_from_working"] + data["tags"],
            "metadata": {
                "promoted": True,
                "original_slot": slot,
                "access_count": data["access_count"],
            }
        }

    # ─── State queries ───────────────────────────────────────────────────────

    def is_empty(self) -> bool:
        """True if no slots are currently written."""
        return len(self._slots) == 0

    def slot_count(self) -> int:
        """Number of active slots."""
        return len(self._slots)

    def summary(self) -> str:
        """One-line human-readable summary of current working memory state."""
        if not self._slots:
            return "Working memory: empty"
        parts = [f"{slot}={data['content'][:40]}{'...' if len(data['content']) > 40 else ''}"
                 for slot, data in list(self._slots.items())[:5]]
        if len(self._slots) > 5:
            parts.append(f"+{len(self._slots) - 5} more")
        return "Working memory: " + " | ".join(parts)
