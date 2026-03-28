"""
Samsara Memory — 5-Layer Cognitive Memory for AI Agents
Forked from mem0 (Apache 2.0) with major extensions.

5 layers:
- Episodic: step-by-step event sequences with tool calls and outcomes
- Semantic: entities, facts, relationships about the world
- Procedural: skills, methods, how-to knowledge
- Working: current context — what's happening now
- Metacognitive: self-model — what the agent knows about itself

License: Apache 2.0 (same as mem0)
"""

from samsara_memory.core import SamsaraMemory
from samsara_memory.layers.episodic import EpisodicMemory
from samsara_memory.layers.semantic import SemanticMemory
from samsara_memory.layers.procedural import ProceduralMemory
from samsara_memory.layers.working import WorkingMemory
from samsara_memory.layers.metacognitive import MetacognitiveMemory
from samsara_memory.types import MemoryLayer, MemoryEntry, TrajectoryEvent

__version__ = "0.1.0"
__all__ = [
    "SamsaraMemory",
    "EpisodicMemory",
    "SemanticMemory",
    "ProceduralMemory",
    "WorkingMemory",
    "MetacognitiveMemory",
    "MemoryLayer",
    "MemoryEntry",
    "TrajectoryEvent",
]
