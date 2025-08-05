import asyncio
import random
from llama_index.core.callbacks import TokenCountingHandler
from llama_index.core.indices.property_graph import PropertyGraphIndex, SchemaLLMPathExtractor
import openai
from llama_index.core.callbacks import CallbackManager
from dotenv import load_dotenv
from langgraph.graph import StateGraph
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from pydantic import BaseModel
from typing import List, Optional
from llama_index.core.schema import Document

# 🧾 Cost rates (adjust if needed)
MODEL_COST = {
    "gpt-4o": {"input": 0.005, "output": 0.015},         # $ per 1K tokens
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "text-embedding-ada-002": {"input": 0.0001},
    "text-embedding-3-large": {"input": 0.00013}  ,
    "gpt-3.5-turbo-0125": {"input": 0.0005, "output": 0.0015},
    # accurate as of mid-2024
}

CHOSEN_MODEL = "gpt-3.5-turbo-0125"
CHOSEN_MODEL_EMBEDDINGS = "text-embedding-ada-002"

def simulate_genotype(effect_allele):
    all_alleles = ['A', 'T', 'C', 'G']
    other_alleles = [a for a in all_alleles if a != effect_allele]
    return f"{random.choice(other_alleles + [effect_allele])}/{random.choice(other_alleles + [effect_allele])}"

def count_effect_allele_copies(genotype, effect_allele):
    return genotype.split('/').count(effect_allele)



import os
import networkx as nx
import matplotlib.pyplot as plt

def build_and_save_mock_marvel_graph():
    G = nx.DiGraph()

    # --- Character → Power relationships ---
    G.add_edge("Wolverine", "Regeneration", relation="possesses_power")
    G.add_edge("Storm", "Weather Control", relation="possesses_power")
    G.add_edge("Professor X", "Telepathy", relation="possesses_power")
    G.add_edge("Magneto", "Magnetism", relation="possesses_power")
    G.add_edge("Cyclops", "Optic Blast", relation="possesses_power")
    G.add_edge("Jean Grey", "Telekinesis", relation="possesses_power")
    G.add_edge("Beast", "Super Strength", relation="possesses_power")
    G.add_edge("Mystique", "Shapeshifting", relation="possesses_power")

    # --- Character → Affiliation relationships ---
    G.add_edge("Wolverine", "X-Men", relation="member_of")
    G.add_edge("Storm", "X-Men", relation="member_of")
    G.add_edge("Professor X", "X-Men", relation="member_of")
    G.add_edge("Cyclops", "X-Men", relation="member_of")
    G.add_edge("Jean Grey", "X-Men", relation="member_of")
    G.add_edge("Beast", "X-Men", relation="member_of")

    G.add_edge("Magneto", "Brotherhood", relation="member_of")
    G.add_edge("Mystique", "Brotherhood", relation="member_of")

    # --- Character → Relationship with another character ---
    G.add_edge("Professor X", "Magneto", relation="friend_and_rival_of")
    G.add_edge("Mystique", "Nightcrawler", relation="mother_of")
    G.add_edge("Jean Grey", "Cyclops", relation="partner_of")

    # --- Character → Gene relationships (has_mutation) ---
    G.add_edge("Wolverine", "Gene X-23", relation="has_mutation")
    G.add_edge("Professor X", "X-Gene", relation="has_mutation")
    G.add_edge("Jean Grey", "Telepathy Mutation", relation="has_mutation")
    G.add_edge("Magneto", "Magnetism Gene", relation="has_mutation")
    G.add_edge("Mystique", "Shapeshift Gene", relation="has_mutation")

    # --- Gene → Power relationships (confers) ---
    G.add_edge("Gene X-23", "Regeneration", relation="confers")
    G.add_edge("X-Gene", "Telepathy", relation="confers")
    G.add_edge("Telepathy Mutation", "Telekinesis", relation="confers")
    G.add_edge("Magnetism Gene", "Magnetism", relation="confers")
    G.add_edge("Shapeshift Gene", "Shapeshifting", relation="confers")

    # --- Visualization ---
    pos = nx.spring_layout(G, seed=42)
    edge_labels = nx.get_edge_attributes(G, 'relation')

    nx.draw(G, pos, with_labels=True, node_color='lightcoral', node_size=2500,
            font_size=9, font_weight='bold', arrows=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='gray')

    plt.title("Mock Marvel Universe Graph")
    plt.tight_layout()
    plt.show()

    # --- Save ---
    os.makedirs("graphs", exist_ok=True)
    nx.write_gml(G, "graphs/marvel_graph.gml")
    print("Graph saved to 'graphs/marvel_graph.gml'")

    return G






##################### 📦 FILTER TRIPLETS / DOCUMENTS BEFORE EMBEDDING #####################



def filter_documents_by_rules(
    docs: List[Document],
    include_keywords: Optional[List[str]] = None,
    exclude_keywords: Optional[List[str]] = None,
    max_documents: Optional[int] = 100
) -> List[Document]:
    """
    Filters a list of LlamaIndex Document objects based on keyword rules and document count.

    Args:
        docs: List of Document(text=...) instances.
        include_keywords: Only keep docs that contain ANY of these keywords (case-insensitive).
        exclude_keywords: Remove docs that contain ANY of these keywords (case-insensitive).
        max_documents: Optional limit to number of docs returned (after filtering). If None or large, returns all.

    Returns:
        Filtered list of Document objects.
    """

    filtered = []

    for doc in docs:
        text = doc.text.lower()

        if include_keywords:
            if not any(kw.lower() in text for kw in include_keywords):
                continue

        if exclude_keywords:
            if any(kw.lower() in text for kw in exclude_keywords):
                continue

        filtered.append(doc)

    # Limit to top-N if needed
    return filtered[:max_documents]


def extract_humanized_triplets_from_graph(graph):
    """
    Converts a NetworkX graph into a list of *human-readable* triplet sentences,
    which are easier for the LLM to parse during path extraction.
    """
    triplets = []

    relation_map = {
        "fights_against": "fights against",
        "member_of": "is a member of",
        "located_in": "is located in",
        "wields": "wields",
        "has_power": "has the power of",
        "created_by": "was created by",
        "enemy_of": "is an enemy of"
    }

    for u, v, data in graph.edges(data=True):
        rel_key = data.get('relation', 'related_to')
        natural_rel = relation_map.get(rel_key, rel_key.replace("_", " "))

        # Human-readable sentence
        sentence = f"The entity {u} {natural_rel} the entity {v}."
        triplets.append(sentence)

    return triplets





def calc_cost(model, prompt=0, completion=0, embed=0, embed_model="text-embedding-ada-002"):
    cost = 0.0

    if model in MODEL_COST:
        m = MODEL_COST[model]
        if "input" in m:
            cost += (prompt / 1000) * m["input"]
        if "output" in m:
            cost += (completion / 1000) * m["output"]

    if embed_model in MODEL_COST:
        em = MODEL_COST[embed_model]
        if "input" in em:
            cost += (embed / 1000) * em["input"]

    return round(cost, 5)


def print_cost_breakdown(handler, model="gpt-3.5-turbo-0125", embed_model="text-embedding-ada-002"):
    prompt_tokens = handler.prompt_llm_token_count
    completion_tokens = handler.completion_llm_token_count
    embed_tokens = handler.total_embedding_token_count

    print(f"📊 Embedding Tokens Used: {embed_tokens}")
    print(f"📨 LLM Prompt Tokens Used: {prompt_tokens}")
    print(f"🧠 LLM Completion Tokens Used: {completion_tokens}")

    total_cost = calc_cost(
        model=model,
        prompt=prompt_tokens,
        completion=completion_tokens,
        embed=embed_tokens,
        embed_model=embed_model
    )

    print(f"💸 Total Estimated Cost: ${total_cost:.5f} USD ({total_cost * 100:.2f}¢)")

QUERY_ROUTING_RULES = {
    "mutation_path": ["gene", "mutation", "power", "confers"],
    "team_lookup": ["team", "x-men", "brotherhood", "avengers"]
}

class MarvelState(BaseModel):
    query: str
    query_type: Optional[str] = "default"
    raw_result: Optional[str] = ""
    final_response: Optional[str] = ""

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



if __name__ == '__main__':
    load_dotenv()

    OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
    openai.api_key = OPENAI_API_KEY

    #########################STEP 1 – BUILD BIOMEDICAL GRAPH ##############################
    graph =     build_and_save_mock_marvel_graph()




    triplet_texts = extract_humanized_triplets_from_graph(graph)
    documents = [Document(text=t) for t in triplet_texts]

    filtered_docs = filter_documents_by_rules(
        documents,
        include_keywords=None,  # Example: ["Thor", "Asgard", "Infinity"]
        exclude_keywords=None,  # Example: ["Hydra", "Red Skull"]
        max_documents=100
    )


    # New: Use GPT‑4.. for reasoning and embedding model 3‑large (or 3‑small)


    # Define this ONCE before LLM + embedding are created
    handler = TokenCountingHandler()
    callback_manager = CallbackManager([handler])

    llm = OpenAI(model=CHOSEN_MODEL, api_key=OPENAI_API_KEY, temperature=0.0, callback_manager=callback_manager)
    embed_model = OpenAIEmbedding(model_name=CHOSEN_MODEL_EMBEDDINGS, api_key=OPENAI_API_KEY,
                                  callback_manager=callback_manager)

    # Step 1: Apply path extraction manually
    print("\n🔄 Running manual path extraction over all docs...")
    extracted_nodes = asyncio.run(
        SchemaLLMPathExtractor(llm=llm, strict=False).acall(filtered_docs, show_progress=True))

    # manual triplet extraction:


    # 2. Build the index properly
    index = PropertyGraphIndex(
        nodes=extracted_nodes,
        embed_model=embed_model,
        llm=llm,
        show_progress=True,
    )

    # 3. Create a query engine
    query_engine = index.as_query_engine(
        include_text=True,
        similarity_top_k=3
    )

    print("\n🚀 Running LangGraph-powered orchestrator")

    orchestrator = MarvelGraphOrchestrator(query_engine)
    test_query = "What gene gives Jean Grey her telekinetic powers?"
    final_state = orchestrator.app.invoke({"query": test_query})

    print_cost_breakdown(handler, model=CHOSEN_MODEL, embed_model=CHOSEN_MODEL_EMBEDDINGS)
    print(final_state["final_response"])

# ✅ Easy

# "Which team is Jean Grey a member of?"
# "What power does Magneto have?"


# ✅ Medium
# "What gene mutation gives Jean Grey her powers?"
# "Which character has the Magnetism Gene?"


# ✅ Complex

# "Which characters are linked to world-threatening events through Vibranium and its misuse?"
# "Trace the mutation and power path that links Mystique to her shapeshifting abilities."




