from config import MODEL_COST


def calc_cost(model, prompt=0, completion=0, embed=0, embed_model="text-embedding-ada-002"):
    """
    Estimate the total cost (in USD) based on token usage and model pricing.

    @param model: The name of the language model used (e.g., "gpt-4o").
    @param prompt: Number of prompt tokens consumed.
    @param completion: Number of completion tokens generated.
    @param embed: Number of tokens used for embeddings.
    @param embed_model: The embedding model used (default is "text-embedding-ada-002").
    @return: Total estimated cost in USD (rounded to 5 decimal places).
    """
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
    """
    Print a breakdown of token usage and the corresponding estimated cost.

    @param handler: A token counting handler with `.prompt_llm_token_count`, `.completion_llm_token_count`, and `.total_embedding_token_count` attributes.
    @param model: The name of the LLM model used (default is "gpt-3.5-turbo-0125").
    @param embed_model: The embedding model used (default is "text-embedding-ada-002").
    @return: None
    """
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

