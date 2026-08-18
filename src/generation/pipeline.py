import os
from src.config import LLM_MODEL
from src.generation.llm import Gemini
from src.retriever import retrieve
from src.retriever.query_parser import parse_query

# Minimum Cohere rerank relevance for a supporting chunk to reach the model.
#
# Measured on real phrasings rather than picked by feel. Formally-worded
# questions score around 0.73, but the patient phrasings this system is for
# score far lower for the same correct chunk: "what should i eat if my liver
# is fatty" peaks at 0.601 and "is hep c curable" at 0.656. A 0.65 cut-off
# therefore threw away correct answers and replied "insufficient information".
RELEVANCE_THRESHOLD = 0.45

# The best chunk is always passed through, whatever it scores. Filtering is
# there to keep near-misses out of the prompt, not to decide whether an answer
# exists - the model's escape hatch does that, and it can read the passage.
ALWAYS_KEEP_TOP = 1


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

    # 2. Retrieve. Pass the ParsedQuery itself, not parsed.dense_query:
    #    retrieve() parses internally, so handing it a rewritten string
    #    would parse twice - two Gemini calls per question, and the BM25
    #    leg would get keywords extracted from an already-rewritten
    #    sentence instead of from what the user actually asked.
    print(f"Retrieving top {top_k} contexts...")
    retrieved_chunks = retrieve(parsed, top_k=top_k)

    # 3. Drop weakly-relevant chunks so the model is not invited to answer from
    #    near-misses. The threshold only applies to `rerank_score`, which is a
    #    0-1 relevance judgement. Retrieval scores are not on that scale - RRF
    #    sums weight/(60+rank) and peaks near 0.021 - so comparing them against
    #    0.65 would discard every chunk and make a correct retrieval look like
    #    a knowledge gap.
    if retrieved_chunks and "rerank_score" in retrieved_chunks[0]:
        filtered_chunks = [
            c for i, c in enumerate(retrieved_chunks)
            if i < ALWAYS_KEEP_TOP or c["rerank_score"] >= RELEVANCE_THRESHOLD
        ]
        print(f"Retrieved {len(retrieved_chunks)} chunks; {len(filtered_chunks)} remain "
              f"after filtering (rerank_score >= {RELEVANCE_THRESHOLD}, "
              f"top {ALWAYS_KEEP_TOP} always kept).")
    else:
        filtered_chunks = retrieved_chunks
        print(f"Retrieved {len(retrieved_chunks)} chunks; no rerank scores, filter skipped.")

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
        for chunk in llm.generate_stream(prompt, system_instruction):
            print(chunk, end="", flush=True)
        print("\n")
    else:
        response = llm.generate(prompt, system_instruction)
        print(response)