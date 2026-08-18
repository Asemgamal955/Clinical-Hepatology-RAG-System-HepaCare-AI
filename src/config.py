import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

# Data Paths
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"


# Models
EMBEDDING_MODEL = "embed-v4.0"
EMBEDDING_DIMENSION = 1024
LLM_MODEL = "gemini-3.1-flash-lite"
