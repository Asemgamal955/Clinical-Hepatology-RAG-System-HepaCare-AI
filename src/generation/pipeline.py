from src.config import LLM_MODEL
from src.generation.llm import Gemini
from src.retriever import retrieve
from src.retriever.query_parser import parse_query


def run_rag(query: str, top_k: int = 5, stream: bool = True):
    """Execute full RAG generation pipeline with query parsing and threshold filtering."""
    print(f"\n[Raw Query]: {query}")

    # 1. Parse and expand query terms
    parsed = parse_query(query)
    print("\n[Query Parsing Breakdown]:")
    print(f"  Dense Query  : {parsed.dense_query}")
    print(f"  Sparse Query : {parsed.sparse_query}")
    print(f"  Expansions   : {parsed.expansions or 'None'}")
    print(f"  LLM Rewritten: {parsed.used_llm}\n")

    # 2. Retrieve top chunks
    print(f"Retrieving top {top_k} contexts...")
    retrieved_chunks = retrieve(parsed.dense_query, top_k=top_k)

    # 3. Filter chunks with score < 0.55
    filtered_chunks = [c for c in retrieved_chunks if c.get('score', 0) >= 0.7]
    print(f"Retrieved {len(retrieved_chunks)} chunks; {len(filtered_chunks)} remain after filtering (score >= 0.7).")

    # 4. Build context block
    context_blocks = []
    for i, c in enumerate(filtered_chunks, 1):
        context_blocks.append(
            f"--- Context [{i}] (Score: {c['score']:.3f}) ---\n"
            f"Source: {c.get('url', 'N/A')} | Path: {c.get('section_path', 'N/A')}\n"
            f"Content:\n{c['text']}"
        )
    context_str = "\n\n".join(context_blocks)

    # 5. System prompt structured using the 4 grounding principles
    system_instruction = (
        "1. ROLE:\n"
        "You are a citation-bound clinical evidence tool specializing in hepatology and liver disease, "
        "not a general advisor. Translate clinical jargon into clear, accessible language.\n\n"
        
        "2. CONTEXT BOUNDARY:\n"
        "Answer ONLY from the provided context passages. Do not use outside medical knowledge, assumptions, "
        "or external facts. State nothing else.\n\n"
        
        "3. OUTPUT FORMAT:\n"
        "Structure every response into these clear sections:\n"
        "- Recommendation / Core Findings: Direct summary answering the query.\n"
        "- Supporting Excerpts & Citations: Key excerpts or quotes from the passages with Context ID/Section.\n\n"
        
        "4. ESCAPE HATCH:\n"
        "If the answer cannot be determined from the provided context chunks, state explicitly: "
        "'The provided context does not contain sufficient information to answer this query.'"
    )

    prompt = f"Context:\n{context_str if context_str else 'No relevant context found.'}\n\nUser Question: {query}\n\nAnswer:"

    llm = Gemini(model=LLM_MODEL, system_instruction=system_instruction)

    print("\n[Gemini Answer]:")
    if stream:
        for chunk in llm.generate_stream(prompt):
            print(chunk, end="", flush=True)
        print("\n")
    else:
        response = llm.generate(prompt)
        print(response)