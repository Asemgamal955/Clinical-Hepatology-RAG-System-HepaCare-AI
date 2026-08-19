from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

# --- LLM Generation ---
class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Prompt text to send to the LLM")
    system_instruction: Optional[str] = Field(None, description="Optional system instructions")
    temperature: float = Field(0.7, description="Generation temperature")
    max_output_tokens: Optional[int] = Field(None, description="Max tokens to generate")
    stream: bool = Field(False, description="Whether to stream the response")

class GenerateResponse(BaseModel):
    text: str = Field(..., description="Generated text from the LLM")

# --- Retrieval ---
class RetrieveRequest(BaseModel):
    query: str = Field(..., description="Search query")
    top_k: int = Field(5, description="Number of results to retrieve")
    config: str = Field("full", description="Retrieval config to use: 'dense', 'hybrid', 'hybrid_parse', 'hybrid_rerank', 'full'")

class ChunkResponse(BaseModel):
    id: str
    text: str
    score: float
    section_path: Optional[str] = None
    url: Optional[str] = None
    corpus: Optional[str] = None
    topic: Optional[str] = None
    heading: Optional[str] = None

class RetrieveResponse(BaseModel):
    results: List[ChunkResponse]

# --- RAG ---
class QueryRequest(BaseModel):
    query: str = Field(..., description="RAG query")
    top_k: int = Field(5, description="Number of context chunks to retrieve")
    stream: bool = Field(False, description="Whether to stream the answer")

class QueryResponse(BaseModel):
    answer: str
    needs_retrieval: bool
    dense_query: Optional[str] = None
    sparse_query: Optional[str] = None
    expansions: Optional[str] = None
    used_llm: bool
    retrieved_chunks: List[ChunkResponse]

# --- Index Status ---
class IndexStatusResponse(BaseModel):
    status: str = Field(..., description="Current status: idle, indexing, success, error")
    progress: float = Field(..., description="Progress percentage from 0 to 100")
    message: str = Field(..., description="Detailed status message")
    error: Optional[str] = Field(None, description="Error details if status is error")

# --- Evaluation ---
class EvaluateRequest(BaseModel):
    config: str = Field("all", description="Config to run: 'all', 'dense', 'hybrid', 'hybrid_parse', 'hybrid_rerank', 'full', 'parse_norawq', 'full_norawq'")
    k: List[int] = Field([5], description="Rank cut-off K values list, e.g. [1, 3, 5, 10]")
    queries_path: str = Field("data/eval/queries.jsonl", description="Path to evaluation queries JSONL file")
    allow_degraded: bool = Field(False, description="Allow degraded fallback under rate limits")

class ConfigSummaryResponse(BaseModel):
    config: str
    precision_at_k: float
    recall_at_k: float
    map_at_k: float
    mrr: float
    seconds_per_query: float
    k: int

class QueryDetailResponse(BaseModel):
    query: str
    retrieved: List[str]
    precision_at_k: float
    recall_at_k: float
    ap_at_k: float
    rr: float

class EvaluateResponse(BaseModel):
    summaries: List[ConfigSummaryResponse]
    details: Dict[str, List[QueryDetailResponse]] = Field(default_factory=dict, description="Detailed per-query results keyed by 'config@k'")

# --- Frontend API Integrations ---
class ChatRequest(BaseModel):
    message: str = Field(..., description="The user message")
    history: Optional[List[Dict[str, Any]]] = Field(None, description="Recent conversation history")

class ChatSourceResponse(BaseModel):
    title: str
    journal: str
    url: str
    authors: Optional[str] = None
    abstract: Optional[str] = None

class RetrievedChunkInfo(BaseModel):
    """Slim chunk info returned to the frontend for display."""
    id: str
    text: str
    score: float
    section_path: Optional[str] = None
    url: Optional[str] = None
    corpus: Optional[str] = None
    heading: Optional[str] = None

class ChatVerificationResponse(BaseModel):
    verdict: str = Field(..., description="Audit verdict: PASSED or FAILED")
    is_grounded: bool = Field(..., description="True if claims are factually grounded")
    citations_valid: bool = Field(..., description="True if cited chunks exist and match")
    no_personalization: bool = Field(..., description="True if response is impersonal and objective")
    certainty: float = Field(..., description="Certainty score from reranker")
    audit_notes: str = Field("", description="Verification summary notes")
    flagged_issues: List[str] = Field(default_factory=list, description="List of violations found")

class ChatResponse(BaseModel):
    reply: str
    source: Optional[ChatSourceResponse] = None
    retrieved_chunks: List[RetrievedChunkInfo] = Field(default_factory=list, description="Evidence chunks retrieved from the RAG pipeline")
    verification: Optional[ChatVerificationResponse] = None

class AssessmentRequest(BaseModel):
    fatigueLevel: int
    painLocation: str
    dietaryAdherence: str
    medications: Optional[str] = None
    recentLabs: Optional[str] = None

class AssessmentResponse(BaseModel):
    wellnessScore: int
    riskLevel: str
    recommendations: List[str]
    timestamp: str

