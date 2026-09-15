from pathlib import Path

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from src.escalation import (
    check_evidence_sufficiency,
    should_escalate_after_response,
    should_escalate_before_response,
)
from src.intents import IntentClassifier
from src.response import ResponseGenerator
from src.retrieval import HistoricalRetriever


class AgentState(TypedDict, total=False):
    message: str

    intent: str
    intent_confidence: float

    evidence: list

    evidence_sufficient: bool
    evidence_reason: str

    reply: str

    grounding_passed: bool
    grounding_reason: str

    decision: str
    reason: str


class SupportAgent:
    def __init__(
        self,
        intent_model_path="data/processed/intent_model.joblib",
        retrieval_directory="data/processed/chroma_db",
    ):
        self.intent_classifier = IntentClassifier()

        model_path = Path(intent_model_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Intent model not found: {model_path}"
            )

        self.intent_classifier.load(model_path)

        self.retriever = HistoricalRetriever(
            persist_directory=retrieval_directory
        )

        self.response_generator = ResponseGenerator()

        self.graph = self._build_graph()

    def classify_intent(self, state):
        result = self.intent_classifier.predict(
            state["message"]
        )

        return {
            "intent": result["intent"],
            "intent_confidence": result["confidence"],
        }

    def retrieve_history(self, state):
        evidence = self.retriever.retrieve(
            state["message"],
            top_k=5,
        )

        return {
            "evidence": evidence,
        }

    def check_evidence(self, state):
        result = check_evidence_sufficiency(
            state.get("evidence", [])
        )

        return {
            "evidence_sufficient": result["sufficient"],
            "evidence_reason": result["reason"],
        }

    def decide_before_response(self, state):
        result = should_escalate_before_response(
            state["intent"],
            {
                "sufficient": state["evidence_sufficient"],
                "reason": state["evidence_reason"],
            },
        )

        if result["decision"] == "ESCALATE":
            return {
                "decision": "ESCALATE",
                "reason": result["reason"],
            }

        return {}

    def route_after_evidence(self, state):
        if state.get("decision") == "ESCALATE":
            return "escalate"

        return "generate"

    def generate_response(self, state):
        reply = self.response_generator.generate(
            customer_message=state["message"],
            intent=state["intent"],
            evidence=state["evidence"],
        )

        return {
            "reply": reply,
        }

    def grounding_check(self, state):
        result = self.response_generator.grounding_check(
            reply=state["reply"],
            customer_message=state["message"],
            evidence=state["evidence"],
        )

        return {
            "grounding_passed": result["passed"],
            "grounding_reason": result["reason"],
        }

    def decide_after_grounding(self, state):
        result = should_escalate_after_response(
            {
                "passed": state["grounding_passed"],
                "reason": state["grounding_reason"],
            }
        )

        return {
            "decision": result["decision"],
            "reason": result["reason"],
        }

    def _build_graph(self):
        graph = StateGraph(AgentState)

        graph.add_node(
            "classify_intent",
            self.classify_intent,
        )

        graph.add_node(
            "retrieve_history",
            self.retrieve_history,
        )

        graph.add_node(
            "check_evidence",
            self.check_evidence,
        )

        graph.add_node(
            "decide_before_response",
            self.decide_before_response,
        )

        graph.add_node(
            "generate_response",
            self.generate_response,
        )

        graph.add_node(
            "grounding_check",
            self.grounding_check,
        )

        graph.add_node(
            "decide_after_grounding",
            self.decide_after_grounding,
        )

        graph.add_node(
            "escalate",
            lambda state: {
                "reply": "",
            },
        )

        graph.add_edge(
            START,
            "classify_intent",
        )

        graph.add_edge(
            "classify_intent",
            "retrieve_history",
        )

        graph.add_edge(
            "retrieve_history",
            "check_evidence",
        )

        graph.add_edge(
            "check_evidence",
            "decide_before_response",
        )

        graph.add_conditional_edges(
            "decide_before_response",
            self.route_after_evidence,
            {
                "generate": "generate_response",
                "escalate": "escalate",
            },
        )

        graph.add_edge(
            "generate_response",
            "grounding_check",
        )

        graph.add_edge(
            "grounding_check",
            "decide_after_grounding",
        )

        graph.add_edge(
            "decide_after_grounding",
            END,
        )

        graph.add_edge(
            "escalate",
            END,
        )

        return graph.compile()

    def predict(self, message):
        if not message or not message.strip():
            raise ValueError(
                "Message cannot be empty."
            )

        result = self.graph.invoke(
            {
                "message": message.strip()
            }
        )

        return {
            "intent": result.get("intent"),
            "confidence": result.get(
                "intent_confidence"
            ),
            "reply": result.get("reply", ""),
            "decision": result.get("decision"),
            "reason": result.get("reason"),
            "evidence": result.get(
                "evidence",
                [],
            ),
        }