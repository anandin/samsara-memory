"""
Core data types for Samsara Memory.
"""
from enum import Enum
from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field


class MemoryLayer(str, Enum):
    """The 5 layers of Samsara memory."""
    EPISODIC = "episodic"           # What happened — step by step
    SEMANTIC = "semantic"           # What is true — facts and entities
    PROCEDURAL = "procedural"       # How to do X — skills and methods
    WORKING = "working"             # What I'm thinking now
    METACOGNITIVE = "metacognitive" # What I know about myself


class ToolCall(BaseModel):
    """A single tool call in a trajectory."""
    tool_name: str
    input: Dict[str, Any]
    output: Optional[str] = None
    latency_ms: float
    step: int
    success: bool = True
    error: Optional[str] = None


class TrajectoryEvent(BaseModel):
    """A complete agent execution episode — stored in episodic memory."""
    session_id: str
    agent_id: str
    task: str                              # What the agent was asked to do
    steps: List[ToolCall]                  # All tool calls in order
    outcome: Literal["success", "failure", "partial"] = "success"
    failure_step: Optional[int] = None      # Which step failed (if any)
    failure_reason: Optional[str] = None    # Why it failed
    duration_ms: float
    model_used: Optional[str] = None
    confidence_before: Optional[float] = None  # 0-1, agent's self-rating before
    confidence_after: Optional[float] = None   # 0-1, agent's self-rating after
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MemoryEntry(BaseModel):
    """A single memory entry in any layer."""
    id: str
    layer: MemoryLayer
    content: str
    agent_id: str
    created_at: str
    updated_at: Optional[str] = None
    version: int = 1
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    # For episodic layer
    trajectory_id: Optional[str] = None
    # For metacognitive layer
    capability: Optional[str] = None        # What capability this relates to
    confidence_score: Optional[float] = None # 0-1


class CapabilityProfile(BaseModel):
    """The agent's model of its own capabilities — lives in metacognitive layer."""
    agent_id: str
    capabilities: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    # structure:
    # {
    #   "refund_processing": {
    #     "confidence": 0.94,
    #     "success_rate": 0.97,
    #     "total_attempts": 142,
    #     "last_success": "2026-03-29T...",
    #     "failure_patterns": ["disputed charges", "cross-border"],
    #     "eval_status": "confidence"  # confident | developing | unknown | declined
    #   }
    # }


class SelfModel(BaseModel):
    """The agent's complete self-model — constructed from metacognitive memory."""
    agent_id: str
    capabilities: Dict[str, CapabilityProfile]
    strengths: List[str] = []
    weaknesses: List[str] = []
    recent_failures: List[str] = []  # failure pattern descriptions
    eval_suite_coverage: int = 0    # number of test cases covering this agent
    overall_health_score: float = 0.0  # 0-1 composite score
