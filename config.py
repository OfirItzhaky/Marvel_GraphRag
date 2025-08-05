import os
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")

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

QUERY_ROUTING_RULES = {
    "mutation_path": ["gene", "mutation", "power", "confers"],
    "team_lookup": ["team", "x-men", "brotherhood", "avengers"]
}