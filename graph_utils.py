
import os
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Optional
from llama_index.core.schema import Document

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


