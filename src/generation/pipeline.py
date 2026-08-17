from src.config import LLM_MODEL
from src.generation.llm import Gemini
from src.retriever import retrieve

def run_rag(query: str, top_k: int = 5, stream: bool = True):
    """Execute full RAG generation pipeline."""
    print(f"\n[Query]: {query}\n")
    print(f"Retrieving top {top_k} contexts...")
    retrieved_chunks = retrieve(query, top_k=top_k)

    # Build context block utilizing section hierarchy
    context_blocks = []
    for i, c in enumerate(retrieved_chunks, 1):
        context_blocks.append(
            f"--- Context [{i}] (Score: {c['score']:.3f}) ---\n"
            f"Source: {c['url']} | Path: {c['section_path']}\n"
            f"Content:\n{c['text']}"
        )
    context_str = "\n\n".join(context_blocks)

    system_instruction = (
        "You are an expert technical AI assistant. Answer the user's question accurately "
        "and concisely using solely the provided context chunks. Cite sources or section "
        "paths where appropriate. If the answer cannot be determined from context, state so."
    )

    prompt = f"Context:\n{context_str}\n\nUser Question: {query}\n\nAnswer:"

    llm = Gemini(model=LLM_MODEL, system_instruction=system_instruction)

    print("\n[Gemini Answer]:")
    if stream:
        for chunk in llm.generate_stream(prompt):
            print(chunk, end="", flush=True)
        print("\n")
    else:
        response = llm.generate(prompt)
        print(response)
