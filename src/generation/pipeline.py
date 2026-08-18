from src.config import LLM_MODEL
from src.generation.llm import Gemini
from src.retriever import retrieve
from src.retriever.query_parser import parse_query  # 1. Import Query Parser


def run_rag(query: str, top_k: int = 5, stream: bool = True):
    """Execute full RAG generation pipeline with query parsing."""
    print(f"\n[Raw Query]: {query}")

    # 2. Parse and expand query terms
    parsed = parse_query(query)
    print("\n[Query Parsing Breakdown]:")
    print(f"  Dense Query  : {parsed.dense_query}")
    print(f"  Sparse Query : {parsed.sparse_query}")
    print(f"  Expansions   : {parsed.expansions or 'None'}")
    print(f"  LLM Rewritten: {parsed.used_llm}\n")

    # 3. Retrieve. Pass the ParsedQuery itself, not parsed.dense_query:
    #    retrieve() parses internally, so handing it a rewritten string
    #    would parse twice - two Gemini calls per question, and the BM25
    #    leg would get keywords extracted from an already-rewritten
    #    sentence instead of from what the user actually asked.
    print(f"Retrieving top {top_k} contexts...")
    retrieved_chunks = retrieve(parsed, top_k=top_k)

    # 4. Build context block utilizing section hierarchy
    context_blocks = []
    for i, c in enumerate(retrieved_chunks, 1):
        context_blocks.append(
            f"--- Context [{i}] (Score: {c['score']:.3f}) ---\n"
            f"Source: {c.get('url', 'N/A')} | Path: {c.get('section_path', 'N/A')}\n"
            f"Content:\n{c['text']}"
        )
    context_str = "\n\n".join(context_blocks)

    system_instruction = (
    "You are an expert medical AI assistant specializing in hepatology and liver disease. "
    "Provide clear, evidence-based information using strictly the provided context chunks. "
    "Translate medical jargon into accessible language while maintaining clinical accuracy. "
    "If the answer cannot be determined from context, state so directly. "
    "Always maintain an empathetic, objective tone and include an educational disclaimer advising "
    "consultation with a healthcare provider for diagnostic or treatment decisions."
    )

    # Use original raw query in the user prompt so generation addresses the exact user question
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