# 🚀 Marvel GraphRAG: Project Gene-Forge

Welcome to **Project Gene-Forge**, a hybrid AI system that combines a knowledge graph of Marvel characters, their genes, powers, and affiliations — with the power of LLMs to generate rich, accurate answers. Built as part of an AI engineering task, this project helps S.H.I.E.L.D. operatives answer questions like:

> "What gene gives Jean Grey her telekinetic powers?"

> "Which team is Wolverine a part of?"

> "Trace the mutation that links Magneto to his powers."

---

## 📊 Overview

This platform merges structured data (a graph of characters, genes, powers, and teams) with natural language reasoning powered by OpenAI models. It includes:

* ✨ **A custom-built Marvel Knowledge Graph** (via NetworkX)
* 🧐 **Path extraction and LLM-based indexing** using LlamaIndex
* 💡 **LangGraph-based orchestration** for query routing and LLM response generation
* 📊 **Token cost tracking** and cache-aware flows
* 📆 **Flask API** with endpoints for queries, graph inspection, and cache control
* 🌐 **Interactive Frontend** for entering questions and viewing results visually

---

## 🚀 Quickstart

### ⚡ Prerequisites

* Python 3.11
* OpenAI API key (you'll enter this via the UI)

### 🔧 Installation

```bash
# Clone the repo
$ git clone https://github.com/OfirItzhaky/Marvel_GraphRag

# Make sure you're in the project directory before running
$ cd Marvel_GraphRag

# Create a virtual environment
$ python -m venv venv
$ source venv/bin/activate  # Or venv\Scripts\activate on Windows

# Install dependencies
$ pip install -r requirements.txt
```
# Setup ENV Variables

✅ Option 1: Set as an Environment Variable (Recommended for security)
# On macOS/Linux
$ export OPENAI_API_KEY=your-api-key-here

# On Windows (CMD)
> set OPENAI_API_KEY=your-api-key-here

# On Windows (PowerShell)
> $env:OPENAI_API_KEY="your-api-key-here"
💡 PyCharm Users:
You can set the environment variable in Run → Edit Configurations → Environment Variables.

⚠️ Option 2: Enter via the UI (less secure)
If you don’t set the API key as an environment variable, the app will prompt you to enter it in the browser UI.
This works — but keep in mind it's less secure, as your key is visible in the request payload.

### 🚀 Run the App

```bash
$ python app.py
```

Then open your browser to `http://localhost:5000`

---

## 🔍 Features

### ✍️ LLM-Powered Q\&A

* Type in any natural-language Marvel question
* System queries the knowledge graph, extracts paths, and calls an LLM to return an answer
* Cost of each call is estimated (tokens + \$)

### 📁 Graph Cache Awareness

* First-time queries generate the graph, triplets, and index (visible in the UI)
* Repeated queries reuse cache to save cost
* A **Reset Cache** button clears all generated artifacts

### 📊 Graph Visualization

* Interactive zoomable graph viewer using Pyvis
* View character-gene-power-team relationships in a single click

---

## 🔹 Project Structure

```
.
├── app.py                 # Flask app with all API endpoints
├── main.py                # Standalone script for backend debugging
├── config.py              # API keys, model selection
├── requirements.txt       # Python dependencies
│
├── graph_utils.py         # Graph construction, filtering, triplet conversion, viz
├── cost_utils.py          # Token + cost tracking
├── cache_utils.py         # File-based cache management
├── marvel_graph_orchestrator.py  # LangGraph orchestration logic
├── state_models.py        # Pydantic model for LangGraph state
│
├── frontend/
│   ├── index.html         # UI with query input, results, cost
│   ├── script.js          # JS to call API, update DOM
│   └── style.css          # Styling and layout
│
├── graphs/                # Cached Marvel graph (GML)
└── cache/                 # LLM-generated triplets and index (if implemented)
```

---

## 🚷 Endpoints

### POST `/query`

* Accepts JSON: `{ "query": "What gene gives Wolverine his powers?", "model": ..., "embedding_model": ..., "api_key": ... }`
* Returns: `{ "response": ..., "cost_usd": ..., "cache_status": ... }`

### GET `/show-graph`

* Renders an interactive HTML graph (or error if not generated yet)

### POST `/reset-cache`

* Deletes all generated triplets, index, and graph files

### GET `/cache-status`

* Returns current state of graph/index/triplet caches

---

## 💡 Notes for Reviewers

* The knowledge graph is built manually in `graph_utils.py`, but can easily be replaced with JSON/CSV imports.
* LlamaIndex's `SchemaLLMPathExtractor` is used for triplet extraction from readable sentences.
* LangGraph routes queries based on keywords to a single path (can be expanded).
* Cost tracking uses OpenAI’s per-model pricing.
* All models and API keys are user-controlled via the UI.
* Cost calculation uses a configurable dictionary (`MODEL_COST`) to estimate $ cost per model/token type.

📌 Prompt Injection & Contextual Guidance
The system injects contextual instructions (e.g., character bios and confidence thresholds) into the prompt before calling the LLM.
Includes example-based formatting to help the model respond accurately and handle low-confidence facts properly.
---

## 🔍 Sample Queries

| Query                                               | Response                                                     |
| --------------------------------------------------- | ------------------------------------------------------------ |
| "What gene gives Jean Grey her telekinetic powers?" | "Telepathy Mutation gives Jean Grey her telekinetic powers." |
| "Which team is Magneto a member of?"                | "Magneto is a member of the Brotherhood."                    |
| "What power does Mystique have?"                    | "Mystique possesses the power of Shapeshifting."             |

---


## 🪄 Future Enhancements

* Redis or in-memory caching for repeated queries
* Neo4j or TigerGraph backend swap
* Confidence scores in the graph
* Character biographies from Wikipedia or Marvel API


