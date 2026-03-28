import uuid
from typing import Dict, List, Any
from samsara_memory.layers.episodic import EpisodicMemory
from samsara_memory.layers.semantic import SemanticMemory
from samsara_memory.layers.procedural import ProceduralMemory
from samsara_memory.layers.working import WorkingMemory
from samsara_memory.layers.metacognitive import MetacognitiveMemory


class SamsaraMemory:
    def __init__(self, agent_id, vector_store=None, embedding_model=None, llm=None, history_db=None):
        self.agent_id = agent_id
        self.session_id = str(uuid.uuid4())
        self._vector_store = vector_store
        self._embedding = embedding_model
        self._llm = llm
        self._history_db = history_db
        if vector_store and embedding_model and llm:
            self._init_layers()

    def _init_layers(self):
        self.episodic = EpisodicMemory(self._vector_store, self._history_db, self._embedding)
        self.semantic = SemanticMemory(self._vector_store, self._embedding, self._llm)
        self.procedural = ProceduralMemory(self._vector_store, self._embedding)
        self.working = WorkingMemory(self._vector_store, self._embedding)
        self.metacognitive = MetacognitiveMemory(self._vector_store, self._embedding)

    def store_trajectory(self, event):
        return self.episodic.store_trajectory(event)

    def recall_similar(self, task, outcome_filter=None):
        return self.episodic.query(self.agent_id, task, outcome_filter=outcome_filter)

    def learn_fact(self, facts, metadata=None):
        return self.semantic.add(self.agent_id, facts, metadata)

    def query_knowledge(self, query):
        return self.semantic.search(self.agent_id, query)

    def register_skill(self, skill_name, procedure, trigger_conditions=None):
        return self.procedural.register_skill(self.agent_id, skill_name, procedure, trigger_conditions)

    def find_skill(self, task):
        return self.procedural.find_skill(self.agent_id, task)

    def push_working(self, content, focus=None, goal=None):
        return self.working.push(self.agent_id, self.session_id, content, focus, goal)

    def get_working_state(self):
        return self.working.get_active(self.agent_id, self.session_id)

    def update_capability(self, capability, success_rate, total_attempts,
                          failure_patterns, eval_status="developing"):
        return self.metacognitive.update_capability(
            self.agent_id, capability, success_rate, total_attempts,
            failure_patterns, eval_status)

    def should_attempt(self, task):
        return self.metacognitive.should_attempt(self.agent_id, task)

    def get_self_model(self):
        return self.metacognitive.get_self_model(self.agent_id)

    def prepare_for_task(self, task):
        should_act, confidence, reason = self.should_attempt(task)
        similar = self.recall_similar(task)
        skills = self.find_skill(task)
        knowledge = self.query_knowledge(task)
        failures = [s for s in similar if s.get("outcome") == "failure"][:3]
        if not should_act:
            rec = "Decline or escalate"
        elif failures:
            rec = f"Caution: {len(failures)} similar failures on record"
        elif skills:
            rec = "Proceed: relevant skills found"
        else:
            rec = "Proceed: no specific risk factors"
        return {
            "task": task, "should_attempt": should_act,
            "self_confidence": confidence, "reason": reason,
            "relevant_skills": skills, "knowledge_base": knowledge,
            "similar_past_experiences": similar, "recent_failures": failures,
            "recommendation": rec,
        }

    def session_summary(self):
        return {
            "agent_id": self.agent_id, "session_id": self.session_id,
            "working_state": self.get_working_state(),
            "self_model": self.get_self_model(),
        }
