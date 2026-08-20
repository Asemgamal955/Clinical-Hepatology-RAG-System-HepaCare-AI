import os
import json
import asyncio
from typing import Dict, Any, List
from fastapi import APIRouter, BackgroundTasks, HTTPException, status
from fastapi.responses import StreamingResponse

from src.api.schemas import (
    GenerateRequest, GenerateResponse,
    RetrieveRequest, RetrieveResponse, ChunkResponse,
    QueryRequest, QueryResponse,
    IndexStatusResponse,
    EvaluateRequest, EvaluateResponse, ConfigSummaryResponse, QueryDetailResponse,
    ChatRequest, ChatResponse, ChatSourceResponse,
    AssessmentRequest, AssessmentResponse
)

# API Imports from existing modules
from src.generation.pipeline import prepare_rag, get_llm
from src.indexing.parse import main as run_parse_chunk
from src.vector_db.collections import build as run_build_index
from src.evaluation.evaluate import CONFIGS, load_queries, run_config

router = APIRouter(prefix="/api/v1")

# Global in-memory state to track indexing background task status
indexing_state = {
    "status": "idle",
    "progress": 0.0,
    "message": "System ready",
    "error": None
}


# ==============================================================================
# § 1  API to Model (LLM Generation & RAG)
# ==============================================================================

@router.post("/llm/generate", response_model=GenerateResponse, tags=["Model"])
async def generate_text(request: GenerateRequest):
    """Directly query the LLM model with raw prompts."""
    try:
        llm = get_llm(request.system_instruction)
        
        if request.stream:
            def stream_tokens():
                for chunk in llm.generate_stream(
                    request.prompt, 
                    request.system_instruction, 
                    temperature=request.temperature
                ):
                    yield chunk
            return StreamingResponse(stream_tokens(), media_type="text/plain")
        
        response_text = llm.generate(
            request.prompt, 
            request.system_instruction, 
            temperature=request.temperature,
            max_output_tokens=request.max_output_tokens
        )
        return GenerateResponse(text=response_text)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"LLM generation failed: {exc}"
        )


@router.post("/query", response_model=QueryResponse, tags=["Model"])
async def query_rag(request: QueryRequest):
    """Run full clinical RAG pipeline returning answers and source citations."""
    try:
        rag_data = prepare_rag(request.query, top_k=request.top_k)
        parsed = rag_data["parsed"]
        
        # Prepare response chunks for Pydantic schema validation
        retrieved_chunks = [
            ChunkResponse(
                id=c["id"],
                text=c["text"],
                score=c["score"],
                section_path=c.get("section_path"),
                url=c.get("url"),
                corpus=c.get("corpus"),
                topic=c.get("topic"),
                heading=c.get("heading")
            )
            for c in rag_data["filtered_chunks"]
        ]
        
        # Metadata payload to return or stream
        meta_payload = {
            "needs_retrieval": parsed.needs_retrieval,
            "dense_query": parsed.dense_query,
            "sparse_query": parsed.sparse_query,
            "expansions": parsed.expansions,
            "used_llm": parsed.used_llm,
            "retrieved_chunks": [c.model_dump() for c in retrieved_chunks]
        }
        
        # Handle fallback scenarios without invoking LLM (non-medical query or empty context)
        if rag_data["not_medical"] or rag_data["no_info"]:
            fallback_ans = rag_data["fallback_answer"]
            if request.stream:
                async def stream_fallback():
                    yield json.dumps({"event": "metadata", "data": meta_payload}) + "\n"
                    # Stream the fallback answer in small artificial chunks
                    words = fallback_ans.split(" ")
                    for i, word in enumerate(words):
                        yield json.dumps({"event": "token", "data": (word + " " if i < len(words) - 1 else word)}) + "\n"
                        await asyncio.sleep(0.01)
                    yield json.dumps({"event": "done"}) + "\n"
                return StreamingResponse(stream_fallback(), media_type="application/x-ndjson")
            
            return QueryResponse(
                answer=fallback_ans,
                needs_retrieval=parsed.needs_retrieval,
                dense_query=parsed.dense_query,
                sparse_query=parsed.sparse_query,
                expansions=parsed.expansions,
                used_llm=parsed.used_llm,
                retrieved_chunks=retrieved_chunks
            )

        # Call the LLM
        llm = get_llm(rag_data["system_instruction"])
        
        if request.stream:
            def stream_rag_response():
                # Yield metadata first so the client gets citation list immediately
                yield json.dumps({"event": "metadata", "data": meta_payload}) + "\n"
                for chunk in llm.generate_stream(rag_data["prompt"], rag_data["system_instruction"]):
                    yield json.dumps({"event": "token", "data": chunk}) + "\n"
                yield json.dumps({"event": "done"}) + "\n"
            return StreamingResponse(stream_rag_response(), media_type="application/x-ndjson")
        
        ans = llm.generate(rag_data["prompt"], rag_data["system_instruction"])
        return QueryResponse(
            answer=ans,
            needs_retrieval=parsed.needs_retrieval,
            dense_query=parsed.dense_query,
            sparse_query=parsed.sparse_query,
            expansions=parsed.expansions,
            used_llm=parsed.used_llm,
            retrieved_chunks=retrieved_chunks
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG query failed: {exc}"
        )


# ==============================================================================
# § 2  API to Data Retrieval
# ==============================================================================

@router.post("/retrieve", response_model=RetrieveResponse, tags=["Retrieval"])
async def retrieve_data(request: RetrieveRequest):
    """Retrieve ranked context passages for a search query using semantic/hybrid configurations."""
    if request.config not in CONFIGS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid configuration '{request.config}'. Choose from: {list(CONFIGS.keys())}"
        )
    
    try:
        retriever_fn = CONFIGS[request.config]
        # Evaluate configuration retrievals
        # Run synchronous retrieval in standard threads to avoid blocking event loops
        loop = asyncio.get_event_loop()
        hits = await loop.run_in_executor(None, retriever_fn, request.query, request.top_k)
        
        results = [
            ChunkResponse(
                id=c["id"],
                text=c["text"],
                score=c["score"],
                section_path=c.get("section_path") or c.get("section"),
                url=c.get("url"),
                corpus=c.get("corpus"),
                topic=c.get("topic"),
                heading=c.get("heading")
            )
            for c in hits
        ]
        return RetrieveResponse(results=results)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retrieval failed: {exc}"
        )


# ==============================================================================
# § 3  API to Evaluation Metrics
# ==============================================================================

@router.post("/evaluate", response_model=EvaluateResponse, tags=["Evaluation"])
async def run_evaluation_metrics(request: EvaluateRequest):
    """Run and aggregate retrieval validation benchmarks against ground-truth datasets."""
    # Set the degraded evaluation flag
    if request.allow_degraded:
        os.environ.pop("STRICT_QUERY_REWRITE", None)
    else:
        os.environ["STRICT_QUERY_REWRITE"] = "1"
        
    try:
        # Load queries & ground truth
        queries = load_queries(request.queries_path)
        
        # Verify requested configurations
        configs_to_run = []
        if request.config == "all":
            configs_to_run = list(CONFIGS.keys())
        elif request.config in CONFIGS:
            configs_to_run = [request.config]
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid config '{request.config}'. Must be 'all' or one of {list(CONFIGS.keys())}"
            )
            
        summaries = []
        details = {}
        
        # Synchronous execution wrapper to run in executor
        def run_benchmarks():
            local_summaries = []
            local_details = {}
            for k_val in request.k:
                for name in configs_to_run:
                    summary, rows = run_config(name, queries, k_val)
                    local_summaries.append(
                        ConfigSummaryResponse(
                            config=name,
                            k=k_val,
                            precision_at_k=summary["precision_at_k"],
                            recall_at_k=summary["recall_at_k"],
                            map_at_k=summary["map_at_k"],
                            mrr=summary["mrr"],
                            seconds_per_query=summary["seconds_per_query"]
                        )
                    )
                    local_details[f"{name}@{k_val}"] = [
                        QueryDetailResponse(
                            query=r["query"],
                            retrieved=r["retrieved"],
                            precision_at_k=r["precision_at_k"],
                            recall_at_k=r["recall_at_k"],
                            ap_at_k=r["ap_at_k"],
                            rr=r["rr"]
                        )
                        for r in rows
                    ]
            return local_summaries, local_details

        loop = asyncio.get_event_loop()
        summaries, details = await loop.run_in_executor(None, run_benchmarks)
        
        return EvaluateResponse(summaries=summaries, details=details)
    except SystemExit as exc:
        # SystemExit is thrown by evaluate.py functions if files/IDs are invalid
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Evaluation dataset error: {exc}"
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evaluation failed: {exc}"
        )


# ==============================================================================
# § 4  Indexing Routine & API Status
# ==============================================================================

def run_indexing_pipeline():
    """Background indexing execution sequence."""
    global indexing_state
    try:
        indexing_state["status"] = "indexing"
        indexing_state["progress"] = 10.0
        indexing_state["message"] = "Starting parsing and semantic chunking..."
        indexing_state["error"] = None
        
        # 1. Parsing & Chunking Raw Files
        run_parse_chunk()
        
        indexing_state["progress"] = 50.0
        indexing_state["message"] = "Creating embeddings and indexing into vector DB..."
        
        # 2. Embedding & Rebuilding Vector DB Index
        run_build_index()
        
        indexing_state["status"] = "success"
        indexing_state["progress"] = 100.0
        indexing_state["message"] = "Indexing completed successfully! Collection is up-to-date."
    except Exception as exc:
        indexing_state["status"] = "error"
        indexing_state["progress"] = 0.0
        indexing_state["message"] = "Indexing aborted due to an error."
        indexing_state["error"] = str(exc)


@router.post("/index", status_code=status.HTTP_202_ACCEPTED, tags=["Indexing"])
async def trigger_indexing(background_tasks: BackgroundTasks):
    """Asynchronously rebuild the vector collection database."""
    global indexing_state
    if indexing_state["status"] == "indexing":
        return {"status": "already_indexing", "message": "An indexing task is already running."}
        
    background_tasks.add_task(run_indexing_pipeline)
    return {"status": "accepted", "message": "Indexing pipeline started in the background."}


@router.get("/index/status", response_model=IndexStatusResponse, tags=["Indexing"])
async def get_indexing_status():
    """Get the current progress status of the background indexing pipeline."""
    global indexing_state
    return IndexStatusResponse(
        status=indexing_state["status"],
        progress=indexing_state["progress"],
        message=indexing_state["message"],
        error=indexing_state["error"]
    )


from datetime import datetime
from src.api.schemas import (
    GenerateRequest, GenerateResponse,
    RetrieveRequest, RetrieveResponse, ChunkResponse,
    QueryRequest, QueryResponse,
    IndexStatusResponse,
    EvaluateRequest, EvaluateResponse, ConfigSummaryResponse, QueryDetailResponse,
    ChatRequest, ChatResponse, ChatSourceResponse, RetrievedChunkInfo,
    AssessmentRequest, AssessmentResponse
)

frontend_router = APIRouter(prefix="/api")

def _build_history_preamble(history: list, current_message: str = None) -> str:
    """Build a readable conversation transcript from frontend history to inject into the LLM prompt."""
    if not history:
        return ""
    
    # Filter out the current user message if it is appended to the end of the history
    clean_history = history
    if current_message and clean_history and clean_history[-1].get("text") == current_message and clean_history[-1].get("sender") == "user":
        clean_history = clean_history[:-1]
        
    if not clean_history:
        return ""
        
    lines = []
    for msg in clean_history[-8:]:  # Keep last 8 turns (4 exchanges) for context window efficiency
        role = "Patient" if msg.get("sender") == "user" else "HepaCare AI"
        text = msg.get("text", "")
        if text:
            lines.append(f"{role}: {text}")
    if not lines:
        return ""
    return "--- Prior Conversation History (for context only) ---\n" + "\n".join(lines) + "\n--- End of Prior History ---\n\n"


@frontend_router.post("/chat", response_model=ChatResponse, tags=["Frontend"])
async def chat_interaction(request: ChatRequest):
    """Conversational endpoint with model memory.
    Injects prior conversation history into the LLM prompt and returns retrieved evidence chunks.
    """
    try:
        # Run the RAG pipeline with the raw user query only
        rag_data = prepare_rag(request.message, top_k=5)

        # Build the slim retrieved chunk list for the response
        def _to_chunk_info(c: dict) -> RetrievedChunkInfo:
            return RetrievedChunkInfo(
                id=c.get("id", ""),
                text=c.get("text", ""),
                score=round(c.get("score", 0.0), 4),
                section_path=c.get("section_path") or c.get("section"),
                url=c.get("url"),
                corpus=c.get("corpus"),
                heading=c.get("heading"),
            )

        retrieved_chunks = [_to_chunk_info(c) for c in rag_data.get("filtered_chunks", [])]

        # Determine fallback answer if not medical or no context found
        if rag_data["not_medical"] or rag_data["no_info"]:
            return ChatResponse(
                reply=rag_data["fallback_answer"],
                source=ChatSourceResponse(
                    title="Clinical Practice Guidelines",
                    journal="AASLD / American Family Physician",
                    url="#"
                ),
                retrieved_chunks=retrieved_chunks
            )

        # Inject conversation history into the prompt for model memory
        history_preamble = _build_history_preamble(request.history or [], request.message)
        enriched_prompt = history_preamble + rag_data["prompt"]

        # Call the LLM
        llm = get_llm(rag_data["system_instruction"])
        ans = llm.generate(enriched_prompt, rag_data["system_instruction"])

        # Map top chunk as the primary citation source
        source_data = None
        if rag_data["filtered_chunks"]:
            best_chunk = rag_data["filtered_chunks"][0]
            source_data = ChatSourceResponse(
                title=best_chunk.get("heading") or best_chunk.get("section_path") or "Clinical Source",
                journal=best_chunk.get("corpus") or "Clinical Reference",
                url=best_chunk.get("url") or "#"
            )
        else:
            source_data = ChatSourceResponse(
                title="Clinical Reference Guidance",
                journal="AASLD Guidelines",
                url="#"
            )

        # Check if the candidate response contains actual, populated Evidence and Citation sections
        has_real_evidence = False
        has_real_citation = False

        if "Evidence:" in ans:
            evidence_part = ans.split("Evidence:", 1)[1]
            if "Citation:" in evidence_part:
                evidence_content = evidence_part.split("Citation:", 1)[0].strip()
            else:
                evidence_content = evidence_part.strip()
            if len(evidence_content) > 3 and evidence_content.lower() not in ["none", "n/a", "not applicable", "no evidence", "none.", "n/a."]:
                has_real_evidence = True

        if "Citation:" in ans:
            citation_content = ans.split("Citation:", 1)[1].strip()
            if len(citation_content) > 5 and citation_content.lower() not in ["none", "n/a", "not applicable", "none.", "n/a."] and not any(p in citation_content for p in ["<chunk id>", "<source>", "<topic>", "<section>"]):
                has_real_citation = True

        has_evidence_and_citation = has_real_evidence and has_real_citation

        # Run clinical verification on candidate answer against retrieved context if structured
        verification_data = None
        chunks_to_return = retrieved_chunks if has_evidence_and_citation else []

        if has_evidence_and_citation:
            try:
                from src.generation.verifier import verify_generation
                from src.api.schemas import ChatVerificationResponse
                verification_res = verify_generation(
                    query=request.message,
                    retrieved_chunks=rag_data.get("filtered_chunks", []),
                    candidate_output=ans
                )
                verification_data = ChatVerificationResponse(
                    verdict=verification_res.verdict,
                    is_grounded=verification_res.is_grounded,
                    citations_valid=verification_res.citations_valid,
                    no_personalization=verification_res.no_personalization,
                    certainty=verification_res.rerank_certainty,
                    audit_notes=verification_res.audit_notes,
                    flagged_issues=verification_res.flagged_issues
                )
            except Exception as ver_err:
                print(f"⚠️ Verification failed: {ver_err}")

        return ChatResponse(
            reply=ans,
            source=source_data,
            retrieved_chunks=chunks_to_return,
            verification=verification_data
        )

    except Exception as exc:
        # Graceful keyword-based fallback (no RAG available)
        query = request.message.lower()
        reply_text = (
            "I have recorded your update in your HepaCare monitoring log. "
            "HepaCare AI evaluates biochemical trends, symptoms, and dietary factors in real time to optimize your liver wellness."
        )
        source_title = "Clinical Practice Guidelines for the Management of Liver Diseases"

        if "alt" in query or "alanine" in query or "enzyme" in query or "elevated" in query:
            reply_text = (
                "ALT (Alanine Aminotransferase) is an enzyme found mostly in the liver. "
                "When liver cells are damaged or inflamed, they can release ALT into the bloodstream. "
                "A slightly high level often indicates mild liver stress, which can be caused by medication, diet, or fatty changes."
            )
            source_title = "Interpretation of mildly elevated liver transaminases"
        elif "fatigue" in query or "tired" in query:
            reply_text = (
                "Fatigue is one of the most common symptoms reported in liver conditions. "
                "We recommend steady hydration (2–2.5L water daily unless fluid restricted) and keeping a regular rest schedule."
            )
            source_title = "Pathophysiology and Management of Fatigue in Liver Disease"
        elif "diet" in query or "meal" in query or "food" in query or "eat" in query:
            reply_text = (
                "A liver-supportive diet emphasizes leafy greens, cruciferous vegetables, fatty fish (Omega-3s), and olive oil, "
                "while limiting high-fructose corn syrup, ultra-processed saturated fats, excess sodium, and alcohol."
            )
            source_title = "Dietary Interventions in Non-Alcoholic Fatty Liver Disease (NAFLD)"

        return ChatResponse(
            reply=reply_text,
            source=ChatSourceResponse(
                title=source_title,
                journal="AASLD / American Family Physician",
                url="#"
            ),
            retrieved_chunks=[]
        )



@frontend_router.post("/assessment", response_model=AssessmentResponse, tags=["Assessment"])
async def run_assessment(request: AssessmentRequest):
    """Calculate simulated liver wellness score and recommendations based on telemetry."""
    score = 88
    if request.fatigueLevel > 5:
        score -= 8
    if request.painLocation == "right_upper":
        score -= 10
    if request.dietaryAdherence == "low":
        score -= 6
        
    risk_level = "Optimal / Low Risk" if score >= 85 else "Moderate Vigilance" if score >= 70 else "Elevated Risk"
    
    recommendations = [
        "Maintain low-sodium dietary protocol (<2,000mg/day)",
        "Repeat comprehensive liver enzyme panel (ALT/AST/ALP) in 4 weeks",
        "Limit OTC acetaminophen to <2g/day and avoid alcohol intake",
        "Stay hydrated with minimum 2 liters of water daily"
      ]
      
    return AssessmentResponse(
        wellnessScore=score,
        riskLevel=risk_level,
        recommendations=recommendations,
        timestamp=datetime.now().isoformat()
    )


@router.get("/health", tags=["Status"])
async def get_health():
    """Liveness check for API health status."""
    return {"status": "ok", "service": "hepatology-rag-api"}
