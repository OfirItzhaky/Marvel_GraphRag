import asyncio

from llama_index.core.callbacks import TokenCountingHandler
from llama_index.core.indices.property_graph import PropertyGraphIndex, SchemaLLMPathExtractor
from llama_index.core.callbacks import CallbackManager

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.schema import Document
from config import OPENAI_API_KEY, CHOSEN_MODEL, CHOSEN_MODEL_EMBEDDINGS
import openai

from cost_utils import print_cost_breakdown
from graph_utils import build_and_save_mock_marvel_graph, extract_humanized_triplets_from_graph, \
    filter_documents_by_rules
from marvel_graph_orchestrator import MarvelGraphOrchestrator

if __name__ == '__main__':

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




