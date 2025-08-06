
from typing import List, Optional
from llama_index.core.schema import Document

import os
import networkx as nx
import matplotlib.pyplot as plt

def build_and_save_mock_marvel_graph(draw=False):
    G = nx.DiGraph()

    # --- Character → Power relationships ---
    G.add_edge("Wolverine", "Regeneration", relation="possesses_power", confidence=0.95)
    G.add_edge("Storm", "Weather Control", relation="possesses_power", confidence=0.95)
    G.add_edge("Professor X", "Telepathy", relation="possesses_power", confidence=0.98)
    G.add_edge("Magneto", "Magnetism", relation="possesses_power", confidence=0.95)
    G.add_edge("Cyclops", "Optic Blast", relation="possesses_power", confidence=0.95)
    G.add_edge("Jean Grey", "Telekinesis", relation="possesses_power", confidence=0.9)
    G.add_edge("Beast", "Super Strength", relation="possesses_power", confidence=0.9)
    G.add_edge("Mystique", "Shapeshifting", relation="possesses_power", confidence=0.9)

    # --- Character → Affiliation relationships ---
    for character in ["Wolverine", "Storm", "Professor X", "Cyclops", "Jean Grey", "Beast"]:
        G.add_edge(character, "X-Men", relation="member_of", confidence=1.0)

    G.add_edge("Magneto", "Brotherhood", relation="member_of", confidence=1.0)
    G.add_edge("Mystique", "Brotherhood", relation="member_of", confidence=1.0)

    # --- Character → Relationship with another character ---
    G.add_edge("Professor X", "Magneto", relation="friend_and_rival_of", confidence=0.8)
    G.add_edge("Mystique", "Nightcrawler", relation="mother_of", confidence=0.65)  # Less certain
    G.add_edge("Jean Grey", "Cyclops", relation="partner_of", confidence=0.7)       # Less certain

    # --- Character → Gene relationships (has_mutation) ---
    G.add_edge("Wolverine", "Gene X-23", relation="has_mutation", confidence=0.95)
    G.add_edge("Professor X", "X-Gene", relation="has_mutation", confidence=0.95)
    G.add_edge("Jean Grey", "Telepathy Mutation", relation="has_mutation", confidence=0.9)
    G.add_edge("Magneto", "Magnetism Gene", relation="has_mutation", confidence=0.9)
    G.add_edge("Mystique", "Shapeshift Gene", relation="has_mutation", confidence=0.9)

    # ✅ Add missing characters with gene mutations and varied confidence:
    G.add_edge("Cyclops", "X-Gene", relation="has_mutation", confidence=0.67)  # Low confidence
    G.add_edge("Storm", "Weather Gene", relation="has_mutation", confidence=0.63)  # Low confidence
    G.add_edge("Beast", "Mutant Strength Gene", relation="has_mutation", confidence=0.88)  # Medium confidence

    # --- Gene → Power relationships (confers) ---
    G.add_edge("Gene X-23", "Regeneration", relation="confers", confidence=0.95)
    G.add_edge("X-Gene", "Telepathy", relation="confers", confidence=0.95)
    G.add_edge("Telepathy Mutation", "Telekinesis", relation="confers", confidence=0.85)
    G.add_edge("Magnetism Gene", "Magnetism", relation="confers", confidence=0.9)
    G.add_edge("Shapeshift Gene", "Shapeshifting", relation="confers", confidence=0.9)

    # --- Save ---
    os.makedirs("graphs", exist_ok=True)
    nx.write_gml(G, "graphs/marvel_graph.gml")
    print("✅ Graph saved to 'graphs/marvel_graph.gml' with confidence scores.")

    if draw:
        draw_graph_with_confidence(G)

    return G


def draw_graph_with_confidence(G):
    pos = nx.spring_layout(G, seed=42)
    edge_labels = {}
    for edge in G.edges:
        rel = G.edges[edge].get('relation', 'related_to')
        confidence = G.edges[edge].get('confidence', 1.0)
        edge_labels[edge] = f"{rel}\n(conf: {confidence:.2f})"
    nx.draw(G, pos, with_labels=True, node_color='lightcoral', node_size=2500,
            font_size=9, font_weight='bold', arrows=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='gray')
    plt.title("Mock Marvel Universe Graph with Confidence Scores")
    plt.tight_layout()
    plt.show()


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

        confidence = data.get('confidence')
        confidence_note = ""
        if confidence is not None and confidence < 0.95:
            confidence_note = f" (confidence: {confidence:.2f})"

        # Final human-readable sentence
        sentence = f"The entity {u} {natural_rel} the entity {v}.{confidence_note}"
        triplets.append(sentence)

    return triplets




