"""
Domain-Agnostic Section-Aware PDF & JSONL Parser & Chunker.

Features:
  - Fully general: No hardcoded publisher names, headings, or stop lists.
  - Layout-aware PDF extraction via PyMuPDF4LLM.
  - General text cleaning: Strips HTML, citations, unglues concatenated words, and fixes unicode.
  - Section-aware: Maintains breadcrumb hierarchy (# H1 > ## H2 > ### H3).
  - Embed-ready JSONL output.

Usage: python -m src.indexing.parse_chunk
"""

import json
import os
import re
import ftfy
import pymupdf4llm
import wordninja

RAW_DIR = os.path.join("data", "raw")
OUT_PATH = os.path.join("data", "processed", "chunks.jsonl")

TARGET_WORDS = 250  # Max words per chunk
OVERLAP_WORDS = 50  # Word overlap between chunks


# ==========================================
# 1. Fully General Text Cleaner
# ==========================================

def clean_text_for_embedding(text: str) -> str:
    """Sanitize raw text extracted from any PDF without publisher-specific rules."""
    if not text:
        return ""

    # 1. Fix encoding and unicode corruptions
    text = ftfy.fix_text(text)

    # 2. Remove HTML / XML tags (e.g. <sup>1,28</sup>)
    text = re.sub(r"<[^>]+>", "", text)

    # 3. Strip standalone URLs and emails
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"\S+@\S+\.\S+", "", text)

    # 4. Remove inline numeric citation artifacts (e.g., "time.1,28" -> "time.")
    text = re.sub(r"(?<=[a-zA-Z.])\d+(?:[,–-]\d+)*(?=\s|[A-Z]|$)", "", text)
    text = re.sub(r"\[\d+(?:[,–-]\d+)*\]", "", text)  # Bracketed [1, 2]

    # 5. Fix Markdown/Math symbol corruptions (e.g., "_ P _ < .001" -> "P < .001")
    text = re.sub(r"_\s*([A-Za-z])\s*_", r"\1", text)

    # 6. Fix concatenated words from PDF font glues (e.g., 'estimatednumberofinfants')
    words = text.split()
    fixed_words = []
    for word in words:
        # If a single word is unusually long (>18 chars) and purely alphabetic
        if len(word) > 18 and word.isalpha():
            fixed_words.append(" ".join(wordninja.split(word)))
        else:
            fixed_words.append(word)
    text = " ".join(fixed_words)

    # 7. Collapse all multi-whitespace and linebreaks into single spaces
    return re.sub(r"\s+", " ", text).strip()


# ==========================================
# 2. Text Chunker & Section Hierarchy
# ==========================================

def split_sentences(text: str) -> list[str]:
    """Split text into sentences cleanly."""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z(])", text) if s.strip()]


def pack(text: str, target: int = TARGET_WORDS, overlap: int = OVERLAP_WORDS) -> list[str]:
    """Greedily pack sentences into chunks with context overlap."""
    chunks, current, count = [], [], 0

    for sentence in split_sentences(text):
        words = len(sentence.split())
        if count + words > target and current:
            chunks.append(" ".join(current))
            tail, tail_words = [], 0
            for prev in reversed(current):
                if tail_words >= overlap:
                    break
                tail.insert(0, prev)
                tail_words += len(prev.split())
            current, count = tail, tail_words

        current.append(sentence)
        count += words

    if current:
        chunks.append(" ".join(current))

    return chunks


def parse_pdf_to_sections(path: str) -> list[dict]:
    """Parse PDF into section blocks using Markdown header hierarchy."""
    md_text = pymupdf4llm.to_markdown(path)

    sections = []
    header_stack = {}  # Tracks level (int) -> heading title (str)
    buffer = []

    current_heading = "Overview"
    current_path = "Overview"

    for line in md_text.split("\n"):
        line_str = line.strip()

        # Parse Markdown headers (# H1, ## H2, ### H3, etc.)
        if line_str.startswith("#"):
            match = re.match(r"^(#+)\s*(.*)", line_str)
            if match:
                hashes, title = match.groups()
                level = len(hashes)
                title = clean_text_for_embedding(title.strip())

                if buffer:
                    sections.append({
                        "heading": current_heading,
                        "section_path": current_path,
                        "body": "\n".join(buffer)
                    })
                    buffer = []

                # Update header tree level
                header_stack = {lvl: h for lvl, h in header_stack.items() if lvl < level}
                header_stack[level] = title

                current_heading = title
                current_path = " > ".join([header_stack[lvl] for lvl in sorted(header_stack.keys())])
        else:
            if line_str:
                buffer.append(line_str)

    if buffer:
        sections.append({
            "heading": current_heading,
            "section_path": current_path,
            "body": "\n".join(buffer)
        })

    return sections


# ==========================================
# 3. Main Pipeline
# ==========================================

def parse_pdf(path: str) -> list[dict]:
    """Extract, clean, and chunk any PDF file."""
    filename = os.path.basename(path)
    doc_stem = os.path.splitext(filename)[0]

    sections = parse_pdf_to_sections(path)
    title = (
        sections[0]["heading"]
        if sections and sections[0]["heading"] != "Overview"
        else doc_stem.replace("_", " ").replace("-", " ").title()
    )

    records = []
    for section in sections:
        clean_body = clean_text_for_embedding(section["body"])

        for piece in pack(clean_body):
            if len(piece.split()) < 20:  # Skip trivial fragments
                continue

            records.append({
                "topic": title,
                "section": section["heading"],
                "section_path": section["section_path"],
                "heading": section["heading"],
                "text": piece,
                "url": filename,
                "source": doc_stem,
                "corpus": "pdf_documents",
            })

    return records


def load_jsonl(path: str) -> list[dict]:
    """Load pre-formatted JSONL datasets."""
    records = []
    corpus_name = os.path.splitext(os.path.basename(path))[0]

    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            row.pop("id", None)
            row["corpus"] = corpus_name

            if "text" in row:
                row["text"] = clean_text_for_embedding(row["text"])

            if "section_path" not in row:
                row["section_path"] = row.get("section") or row.get("heading") or "Overview"

            records.append(row)

    return records


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    records = []

    if not os.path.exists(RAW_DIR):
        print(f"Directory '{RAW_DIR}' not found.")
        return

    for name in sorted(os.listdir(RAW_DIR)):
        file_path = os.path.join(RAW_DIR, name)

        if name.lower().endswith(".jsonl"):
            rows = load_jsonl(file_path)
            records.extend(rows)
            print(f"{name}: {len(rows)} chunks (JSONL)")

        elif name.lower().endswith(".pdf"):
            rows = parse_pdf(file_path)
            records.extend(rows)
            print(f"{name}: {len(rows)} chunks (PDF)")

    # Global unique ID assignment
    for n, record in enumerate(records):
        record["id"] = f"{record['corpus']}-{n:04d}"

    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    total_words = sum(len(r["text"].split()) for r in records)
    print(f"\nTotal: {len(records)} chunks, ~{total_words:,} words -> {OUT_PATH}")


if __name__ == "__main__":
    main()