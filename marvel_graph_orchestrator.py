import json

from langgraph.graph import StateGraph

from character_bios import CHARACTER_BIOS
from config import QUERY_ROUTING_RULES, CHOSEN_MODEL
from state_models import MarvelState


class MarvelGraphOrchestrator:
    def __init__(self, query_engine):
        """
        Initialize the orchestrator with a query engine and build the LangGraph graph.

        @param query_engine: An object that supports `.query()` method for querying the graph.
        """
        self.query_engine = query_engine
        self.app = self._build_graph()


    def classify_query_node(self, state: MarvelState) -> dict:
        """
        Classify the incoming query into a type based on routing rules.

        @param state: The current state containing the user's query.
        @return: A dictionary with updated state including 'query_type'.
        """
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
        """
        Query the graph using the current query and store the raw result in the state.

        @param state: The current MarvelState containing the query.
        @return: A dictionary with the updated state including 'raw_result'.
        """
        print("📡 query_graph_node running...")

        response = self.query_engine.query(state.query)
        state.raw_result = str(response)
        return dict(state)

    def format_response_node(self, state: MarvelState) -> dict:
        """
        Format the final response to be returned to the user.

        @param state: The current MarvelState containing the raw result.
        @return: A dictionary with the updated state including 'final_response'.
        """
        print("🎨 format_response_node running...")

        state.final_response = f"🧠 Answer: {state.raw_result.strip()}"
        return dict(state)

    def _build_graph(self):
        """
        Build and compile the LangGraph workflow graph.

        @return: The compiled graph application object.
        """
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

    def build_modified_prompt(self, user_question: str, bios_dict: dict) -> str:
        """
        Construct a prompt that includes bios, examples, and graph-based instructions for the LLM.

        @param user_question: The raw question from the user.
        @param bios_dict: A dictionary mapping character names to their biography strings.
        @return: A fully formatted prompt string to pass to the LLM.
        """
        bios_snippets = "\n".join([f"{k}: {v}" for k, v in bios_dict.items()])

        example_bio_1 = (
            "Jean Grey: Jean Grey is a powerful mutant telepath and telekinetic. "
            "She shares a deep, often tragic love with Cyclops, and her inner battles have placed her at the center of multiple pivotal events."
        )
        example_bio_2 = (
            "Storm: Storm is a mutant with the ability to manipulate weather. "
            "She is a strong leader in the X-Men and often serves as a moral compass for the team."
        )

        example_question_1 = "What gene gives Jean Grey her telekinetic powers?"
        example_answer_1 = (
            "🧬 Biography:\n"
            f"{example_bio_1}\n\n"
            "🧠 Answer:\n"
            "Jean Grey's telekinetic powers are conferred by the Telepathy Mutation."
        )

        example_question_2 = "What gene gives Storm her weather powers?"
        example_answer_2 = (
            "🧬 Biography:\n"
            f"{example_bio_2}\n\n"
            "🧠 Answer:\n"
            "Storm's weather control powers are possibly linked to the Weather Gene (confidence: 0.63)."
        )

        confidence_instruction = (
            "⚠️ IMPORTANT: Some graph facts include a numeric confidence score.\n"
            "If you reference a fact with confidence below **0.8**, you MUST:\n"
            "- Use uncertainty phrasing like: \"possibly\", \"is likely\", or \"with some uncertainty\"\n"
            "- Include the numeric score: e.g., (confidence: 0.67)\n\n"
            "✅ Examples of accepted phrasing:\n"
            "\"Storm's powers are possibly linked to the Weather Gene (confidence: 0.63).\"\n"
            "\"Jean Grey may have the Telepathy Mutation (confidence: 0.72).\"\n\n"
            "❌ Do NOT state the fact confidently or omit the score if it's below 0.8."
        )

        return (
            "You are a Marvel AI assistant trained on genetic data, powers, and affiliations.\n"
            "When answering questions, follow these instructions strictly:\n"
            "1. Identify the characters mentioned or implied by the question.\n"
            "2. For each such character, you MUST begin your answer with their biography from the list below.\n"
            "   - If a biography is not available, say: \"⚠️ Biography for [Character Name] not found in this database.\"\n"
            "   - You MUST then still proceed to answer the user’s question using only the available graph facts.\n"
            "3. Avoid speculation or outside knowledge.\n\n"
            f"{confidence_instruction}\n\n"
            "Bios:\n"
            f"{bios_snippets}\n\n"
            "Here are two example formats:\n"
            f"Q: {example_question_1}\n"
            f"{example_answer_1}\n\n"
            f"Q: {example_question_2}\n"
            f"{example_answer_2}\n\n"
            f"Now answer this question:\nQ: {user_question}"
        )

