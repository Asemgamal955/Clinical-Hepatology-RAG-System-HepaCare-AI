"""
Parse and chunk the raw corpus into a single embed-ready file.

Inputs (data/raw):
  niddk_liver.jsonl  - NIDDK pages, already split one chunk per <h2> question
  *.pdf              - USPSTF recommendation statements as published in JAMA

Output:
  data/processed/chunks.jsonl  - unified schema, one chunk per line

The two sources need different treatment. NIDDK is already chunked at a
natural boundary, so it is only re-tagged. The PDFs are two-column journal
articles, so they need the journal furniture stripped, the line-wrap
artifacts repaired, and then splitting by section and size.

Usage:  python -m src.indexing.parse_chunk
"""

import json
import os
import re

from pypdf import PdfReader

RAW_DIR = os.path.join("data", "raw")
OUT_PATH = os.path.join("data", "processed", "chunks.jsonl")

TARGET_WORDS = 250  # upper bound per PDF chunk
OVERLAP_WORDS = 50  # carried into the next chunk so sentences aren't orphaned

# Section headings used by USPSTF recommendation statements. Matching against a
# known list beats a generic "short title-case line" rule, which on these PDFs
# also fires on "Editorial page 312", "Stanford, CA 94305-6019" and stray
# drop-cap letters.
USPSTF_HEADINGS = [
    "Summary of Recommendation and Evidence",
    "Summary of Recommendation",
    "Rationale",
    "Importance",
    "Detection",
    "Benefits of Early Detection and Intervention and Treatment",
    "Benefits of Early Detection and Treatment",
    "Harms of Early Detection and Intervention and Treatment",
    "Harms of Early Detection and Treatment",
    "USPSTF Assessment",
    "Assessment of Magnitude of Net Benefit",
    "Practice Considerations",
    "Patient Population Under Consideration",
    "Assessment of Risk",
    "Screening Tests",
    "Screening Intervals",
    "Screening Implementation",
    "Treatment or Interventions",
    "Treatment and Interventions",
    "Treatment",
    "Additional Approaches to Prevention",
    "Additional Tools and Resources",
    "Other Related USPSTF Recommendations",
    "Discussion",
    "Burden of Disease",
    "Scope of Review",
    "Accuracy of Screening Tests",
    "Accuracy of Screening Tests and Risk Assessment",
    "Effectiveness of Early Detection and Treatment",
    "Benefits of Early Detection or Treatment",
    "Potential Harms of Screening and Treatment",
    "Harms of Screening or Treatment",
    "Estimate of Magnitude of Net Benefit",
    "Response to Public Comment",
    "Research Needs and Gaps",
    "Recommendations of Others",
    "Update of Previous USPSTF Recommendation",
    "Supporting Evidence",
]

# Everything from here on is boilerplate, not content worth retrieving.
STOP_HEADINGS = [
    "ARTICLE INFORMATION",
    "Conflict of Interest Disclosures",
    "REFERENCES",
    "References",
]

# Running heads, copyright lines, and the marginal "see also" box on page 1.
JUNK_PATTERNS = [
    r"^\d{1,3}$",
    r"^©\s*\d{4}\s*American Medical Association",
    r"^Clinical Review & Education",
    r"^JAMA\b.*doi:",
    r"^Downloaded [Ff]rom",
    r"^(Editorial|Related article|Author Audio|Audio and Supplemental|"
    r"JAMA Patient Page|CME Quiz|Related articles)\b",
    r"^Author/Group Information",
    r"^(Accepted for Publication|Corresponding Author|Published Online)",
    r"^\(Reprinted\)",
    r"^jama\.com",
    r"Mail Code|Stanford, CA",
]
JUNK_RE = re.compile("|".join(JUNK_PATTERNS))

# A superscript reference marker that extraction dropped onto its own line,
# e.g. "1,2 Although there are guidelines ...".
LEADING_CITE_RE = re.compile(r"^\d{1,3}(?:[,–-]\d{1,3})*\s+(?=[A-Z(“])")
TRAILING_CITE_RE = re.compile(r"(?<=[.,;])\s+\d{1,3}(?:[,–-]\d{1,3})*\s*$")

# Prefixes where the hyphen is part of the word, not a line-wrap artifact.
KEEP_HYPHEN = ("non", "self", "pre", "post", "anti", "co", "multi", "inter", "intra")


def clean_lines(raw_text):
    """Drop journal furniture and reference markers, line by line."""
    kept = []
    for line in raw_text.split("\n"):
        line = line.strip()
        if not line or JUNK_RE.search(line):
            continue
        line = LEADING_CITE_RE.sub("", line)
        line = TRAILING_CITE_RE.sub("", line)
        if line:
            kept.append(line)

    # A drop cap extracts as its own line, leaving "T" above "he US Preventive
    # Services Task Force...". Glue it back on; drop other stray letters
    # (grade markers such as a lone "C") instead.
    out = []
    for i, line in enumerate(kept):
        if len(line) == 1 and line.isalpha():
            nxt = kept[i + 1] if i + 1 < len(kept) else ""
            if line.isupper() and nxt[:1].islower():
                kept[i + 1] = line + nxt
            continue
        out.append(line)
    return out


def dehyphenate(text):
    """Repair words split across a line break: 'rec- ommendations' -> one word."""

    def join(match):
        head, tail = match.group(1), match.group(2)
        if head.lower().endswith(KEEP_HYPHEN):
            return f"{head}-{tail}"
        return head + tail

    return re.sub(r"(\w+)-\s+(\w+)", join, text)


def split_sections(lines):
    """Group cleaned lines under their USPSTF heading."""
    headings = {h.lower(): h for h in USPSTF_HEADINGS}
    stops = {s.lower() for s in STOP_HEADINGS}

    sections = []
    current, buffer = "Overview", []
    for line in lines:
        key = line.rstrip(":").lower()
        if key in stops:
            break
        if key in headings:
            if buffer:
                sections.append((current, " ".join(buffer)))
            current, buffer = headings[key], []
        else:
            buffer.append(line)
    if buffer:
        sections.append((current, " ".join(buffer)))
    return sections


def split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z(])", text) if s.strip()]


def pack(text, target=TARGET_WORDS, overlap=OVERLAP_WORDS):
    """Greedily pack sentences into chunks, carrying an overlap tail forward."""
    chunks, current, count = [], [], 0
    for sentence in split_sentences(text):
        words = len(sentence.split())
        if count + words > target and current:
            chunks.append(" ".join(current))
            tail, tail_words = [], 0
            for prev in reversed(current):  # keep the last ~overlap words
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


def parse_pdf(path):
    reader = PdfReader(path)
    lines = []
    for page in reader.pages:
        lines.extend(clean_lines(page.extract_text() or ""))

    title = lines[0] if lines else os.path.basename(path)
    records = []
    for section, body in split_sections(lines):
        body = dehyphenate(body)
        body = re.sub(r"\s+", " ", body).strip()
        for piece in pack(body):
            if len(piece.split()) < 20:  # stray fragments, not worth indexing
                continue
            records.append(
                {
                    "topic": title,
                    "section": section,
                    "heading": section,
                    "text": piece,
                    "url": os.path.basename(path),
                    "source": "USPSTF recommendation statement (JAMA)",
                    "corpus": "uspstf",
                }
            )
    return records


def load_niddk(path):
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            row.pop("id", None)  # reassigned across the merged corpus
            row["corpus"] = "niddk"
            records.append(row)
    return records


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    records = []

    niddk_path = os.path.join(RAW_DIR, "niddk_liver.jsonl")
    if os.path.exists(niddk_path):
        rows = load_niddk(niddk_path)
        records.extend(rows)
        print(f"niddk_liver.jsonl: {len(rows)} chunks (already chunked, re-tagged)")

    for name in sorted(os.listdir(RAW_DIR)):
        if not name.lower().endswith(".pdf"):
            continue
        rows = parse_pdf(os.path.join(RAW_DIR, name))
        records.extend(rows)
        print(f"{name}: {len(rows)} chunks")

    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        for n, record in enumerate(records):
            record["id"] = f"{record['corpus']}-{n:04d}"
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    words = sum(len(r["text"].split()) for r in records)
    print(f"\n{len(records)} chunks, ~{words:,} words -> {OUT_PATH}")


if __name__ == "__main__":
    main()
