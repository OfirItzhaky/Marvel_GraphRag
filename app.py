from flask import Flask, request, jsonify
import os
import json
from cache_utils import clear_cache, cache_exists, ensure_cache_dir
from config import OPENAI_API_KEY, CHOSEN_MODEL, CHOSEN_MODEL_EMBEDDINGS
from cost_utils import calc_cost
from graph_utils import build_and_save_mock_marvel_graph, extract_humanized_triplets_from_graph, filter_documents_by_rules
from marvel_graph_orchestrator import MarvelGraphOrchestrator
from llama_index.core.callbacks import TokenCountingHandler
from llama_index.core.indices.property_graph import PropertyGraphIndex, SchemaLLMPathExtractor
from llama_index.core.callbacks import CallbackManager
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.schema import Document
import asyncio
from flask import send_from_directory



app = Flask(__name__)

@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    user_query = data.get('query', '')
    if not user_query:
        return jsonify({'error': 'Missing query'}), 400

    # --- Cache status flags ---
    graph_path = os.path.join('graphs', 'marvel_graph.gml')
    triplets_path = os.path.join('cache', 'triplets.json')
    index_path = os.path.join('cache', 'index.json')
    graph_cached = os.path.exists(graph_path)
    triplets_cached = os.path.exists(triplets_path)
    index_cached = os.path.exists(index_path)

    # Step 1: Build/load graph
    if graph_cached:
        import networkx as nx
        graph = nx.read_gml(graph_path)
        graph_status = 'cached'
    else:
        graph = build_and_save_mock_marvel_graph()
        graph_status = 'rebuilt'

    # Step 2: Build/load triplets
    if triplets_cached:
        with open(triplets_path, 'r', encoding='utf-8') as f:
            triplet_texts = json.load(f)
        triplets_status = 'cached'
    else:
        triplet_texts = extract_humanized_triplets_from_graph(graph)
        os.makedirs('cache', exist_ok=True)
        with open(triplets_path, 'w', encoding='utf-8') as f:
            json.dump(triplet_texts, f)
        triplets_status = 'rebuilt'

    documents = [Document(text=t) for t in triplet_texts]
    filtered_docs = filter_documents_by_rules(
        documents,
        include_keywords=None,
        exclude_keywords=None,
        max_documents=100
    )

    # Step 3: Setup LLM, embedding, and callback manager
    handler = TokenCountingHandler()
    callback_manager = CallbackManager([handler])
    llm = OpenAI(model=CHOSEN_MODEL, api_key=OPENAI_API_KEY, temperature=0.0, callback_manager=callback_manager)
    embed_model = OpenAIEmbedding(model_name=CHOSEN_MODEL_EMBEDDINGS, api_key=OPENAI_API_KEY, callback_manager=callback_manager)

    # Step 4: Path extraction
    extracted_nodes = asyncio.run(
        SchemaLLMPathExtractor(llm=llm, strict=False).acall(filtered_docs, show_progress=False)
    )

    # Step 5: Build/load index
    if index_cached:
        with open(index_path, 'r', encoding='utf-8') as f:
            # For demonstration, we just note the cache; actual index loading would require more logic
            pass
        index_status = 'cached'
    else:
        # Save a dummy file to indicate index was built (real index caching would be more complex)
        os.makedirs('cache', exist_ok=True)
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write('built')
        index_status = 'rebuilt'

    index = PropertyGraphIndex(
        nodes=extracted_nodes,
        embed_model=embed_model,
        llm=llm,
        show_progress=False,
    )
    query_engine = index.as_query_engine(
        include_text=True,
        similarity_top_k=3
    )

    # Step 6: Run orchestrator
    orchestrator = MarvelGraphOrchestrator(query_engine)
    final_state = orchestrator.app.invoke({"query": user_query})
    response = final_state["final_response"]

    # Step 7: Calculate cost
    prompt_tokens = handler.prompt_llm_token_count
    completion_tokens = handler.completion_llm_token_count
    embed_tokens = handler.total_embedding_token_count
    cost_usd = calc_cost(
        model=CHOSEN_MODEL,
        prompt=prompt_tokens,
        completion=completion_tokens,
        embed=embed_tokens,
        embed_model=CHOSEN_MODEL_EMBEDDINGS
    )

    build_status = {
        "graph": graph_status,
        "triplets": triplets_status,
        "index": index_status
    }

    return jsonify({
        "response": response,
        "cost_usd": cost_usd,
        "build_status": build_status
    })

@app.route('/reset-cache', methods=['POST'])
def reset_cache():
    clear_cache()
    return jsonify({"status": "cache cleared"})

@app.route('/cache-status', methods=['GET'])
def cache_status():
    # Check for key files
    graph_exists = os.path.exists(os.path.join('graphs', 'marvel_graph.gml'))
    # You can expand this to check for other cache artifacts as needed
    # For now, just a simple example
    cache_files = os.listdir('cache') if os.path.exists('cache') else []
    index_exists = any('index' in f for f in cache_files)
    triplets_exists = any('triplet' in f for f in cache_files)
    return jsonify({
        "graph": graph_exists,
        "index": index_exists,
        "triplets": triplets_exists
    })

@app.route('/')
def serve_index():
    return send_from_directory('frontend', 'index.html')

@app.route('/<path:path>')
def serve_static_file(path):
    return send_from_directory('frontend', path)

if __name__ == '__main__':
    app.run(debug=True)
