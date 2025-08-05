import json

from langgraph.graph import StateGraph

from character_bios import CHARACTER_BIOS
from config import QUERY_ROUTING_RULES, CHOSEN_MODEL
from state_models import MarvelState


class MarvelGraphOrchestrator:
    def __init__(self, query_engine):
        self.query_engine = query_engine
        self.app = self._build_graph()


    def classify_query_node(self, state: MarvelState) -> dict:
        print("🧭 classify_query_node received:", state)

        q = state.query.lower()
        matched_type = "default"

        for query_type, keywords in QUERY_ROUTING_RULES.items():
            if any(k in q for k in keywords):
                matched_type = query_type
                break

        new_state = {
            "query": state.query,
            "query_type": matched_type,
            "raw_result": "",
            "final_response": ""
        }

        print("✅ classify_query_node returning:", new_state)
        return new_state

    def query_graph_node(self, state: MarvelState) -> dict:
        print("📡 query_graph_node running...")

        response = self.query_engine.query(state.query)
        state.raw_result = str(response)
        return dict(state)

    def format_response_node(self, state: MarvelState) -> dict:
        print("🎨 format_response_node running...")

        state.final_response = f"🧠 Answer: {state.raw_result.strip()}"
        return dict(state)

    def _build_graph(self):
        graph = StateGraph(MarvelState)

        graph.add_node("classify", self.classify_query_node)
        graph.add_node("query_graph", self.query_graph_node)
        graph.add_node("format_response", self.format_response_node)

        def route(state: MarvelState) -> str:
            return "query_graph"

        graph.set_entry_point("classify")
        graph.add_conditional_edges("classify", route)
        graph.add_edge("query_graph", "format_response")
        graph.set_finish_point("format_response")

        return graph.compile()