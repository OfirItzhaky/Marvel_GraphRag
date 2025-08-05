from config import MODEL_COST


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

