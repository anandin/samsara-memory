"""
Samsara Memory — 5-Layer Cognitive Memory for Agents

from samsara_memory import SamsaraMemory

memory = SamsaraMemory(agent_id="my-agent")

# Before attempting a task — Samsara Loop gate
eval = memory.self_evaluate(capability="refund_processing")
if not eval["can_attempt"]:
    print(f"Declining: {eval['reason']}")

# After success or failure
memory.record_attempt(capability="refund_processing", success=True)

# When something goes wrong
memory.capture_error(trajectory=steps, error_message=str(e), task="process refund")

# When a human corrects you
memory.capture_correction(human_feedback="You should have escalated this", original_action="processed anyway")

# Search across all layers
results = memory.recall("refund disputes", layers=["episodic", "semantic"])
"""
from samsara_memory.core import SamsaraMemory
from samsara_memory.types import (
    MemoryEntry,
    MemoryLayer,
    TrajectoryEvent,
    ToolCall,
    CapabilityProfile,
    SelfModel,
)

__version__ = "0.1.0"
__all__ = [
    "SamsaraMemory",
    "MemoryEntry",
    "MemoryLayer",
    "TrajectoryEvent",
    "ToolCall",
    "CapabilityProfile",
    "SelfModel",
]
