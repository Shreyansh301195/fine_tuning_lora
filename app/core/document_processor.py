"""
app/core/document_processor.py
───────────────────────────────
Downloads UIDAI policy documents from uidai.gov.in,
parses PDFs, chunks text, and produces JSONL fine-tuning datasets
in Gemma chat-template format.

Pipeline:
  1. Download PDFs → data/raw/
  2. Extract + clean text → data/processed/
  3. Chunk with overlap → instruction/response pairs
  4. Split 80/20 train/valid → data/train/train.jsonl, valid.jsonl
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generator, List, Optional, Tuple

import pdfplumber
import requests
from loguru import logger
from tqdm import tqdm

from app.core.config import settings


# ─────────────────────────────────────────────────────────────────────────────
# UIDAI Document Registry
# ─────────────────────────────────────────────────────────────────────────────

UIDAI_DOCUMENTS = [
    {
        "name": "Aadhaar Act 2016",
        "url": "https://uidai.gov.in/images/targeted_delivery_of_financial_and_other_subsidies_benefits_and_services_13072016.pdf",
        "filename": "aadhaar_act_2016.pdf",
        "description": "The Aadhaar (Targeted Delivery of Financial and Other Subsidies, Benefits and Services) Act, 2016",
    },
    {
        "name": "Aadhaar Authentication Regulation 2016",
        "url": "https://uidai.gov.in/images/regulation/aadhaar_authentication_regulations_2016.pdf",
        "filename": "aadhaar_authentication_regulation_2016.pdf",
        "description": "Regulations governing Aadhaar authentication ecosystem",
    },
    {
        "name": "Aadhaar Data Vault Circular",
        "url": "https://uidai.gov.in/images/news/Circular_11012019.pdf",
        "filename": "aadhaar_data_vault_circular.pdf",
        "description": "UIDAI circular on Aadhaar Data Vault implementation",
    },
    {
        "name": "Aadhaar Enrollment & Update Regulations 2016",
        "url": "https://uidai.gov.in/images/regulation/aadhaar_enrolment_and_update_regulations_2016.pdf",
        "filename": "aadhaar_enrollment_update_2016.pdf",
        "description": "Regulations for biometric enrollment and Aadhaar updates",
    },
    {
        "name": "UIDAI Sharing of Data Policy",
        "url": "https://uidai.gov.in/images/regulation/uidai_sharing_of_data_policy.pdf",
        "filename": "uidai_data_sharing_policy.pdf",
        "description": "Policy document on sharing Aadhaar data and privacy obligations",
    },
]

# Supplementary QA pairs synthesized from UIDAI public website
STATIC_QA_PAIRS: List[Tuple[str, str]] = [
    (
        "What is UIDAI?",
        "The Unique Identification Authority of India (UIDAI) is a statutory authority established under the Aadhaar Act, 2016. It is responsible for enrolling residents, assigning 12-digit unique Aadhaar numbers, and maintaining the Central Identities Data Repository (CIDR).",
    ),
    (
        "What is an Aadhaar number?",
        "An Aadhaar number is a 12-digit unique identification number issued by UIDAI to residents of India. It is based on biometric and demographic data collected during enrollment. The number is permanent and non-transferable.",
    ),
    (
        "What are the privacy protections under the Aadhaar Act 2016?",
        "Section 28 of the Aadhaar Act prohibits sharing of identity information. Biometric information shall not be used for any purpose other than generation and authentication of Aadhaar numbers. UIDAI must ensure confidentiality, integrity, and safety of all identity information.",
    ),
    (
        "What is Aadhaar authentication?",
        "Aadhaar authentication is the process by which Aadhaar number along with biometric or demographic data submitted by a resident is matched against data stored in CIDR, and a Yes/No response is returned by UIDAI to the requesting entity (AUA/KUA).",
    ),
    (
        "What is a Requesting Entity (RE) under UIDAI regulations?",
        "A Requesting Entity (RE) is an agency or person that uses Aadhaar for establishing the identity of an individual. They must register with UIDAI, sign agreements to protect data, and only use Aadhaar authentication for the specific purpose declared to UIDAI.",
    ),
    (
        "What are the obligations of an Authentication User Agency (AUA)?",
        "An AUA must: (1) register with UIDAI; (2) use authentication only for declared purposes; (3) not store biometric data at any point; (4) maintain logs for six months; (5) implement UIDAI-prescribed security standards; and (6) ensure the resident's consent is obtained before authentication.",
    ),
    (
        "Can Aadhaar biometric data be stored by third parties?",
        "No. Under Section 29 and the Aadhaar Authentication Regulations 2016, biometric information including fingerprints, iris scan, and facial photograph cannot be stored, used, or transmitted by any entity other than UIDAI. Violating this is a punishable offence under Section 40 of the Aadhaar Act.",
    ),
    (
        "What is Virtual ID (VID) in Aadhaar?",
        "Virtual ID (VID) is a temporary, revocable 16-digit number mapped to an Aadhaar number. Residents can use VID instead of their Aadhaar number for authentication, protecting their actual Aadhaar number from disclosure. VIDs can be regenerated by the resident at any time.",
    ),
    (
        "What is the Aadhaar Data Vault?",
        "UIDAI mandated through a 2018 circular that all entities storing Aadhaar numbers must do so in an Aadhaar Data Vault — an encrypted, access-controlled repository. The Aadhaar number is replaced with a reference key in all application databases, reducing exposure risk.",
    ),
    (
        "What is the penalty for unauthorized use of Aadhaar identity information?",
        "Under Section 40 of the Aadhaar Act 2016, unauthorized use or disclosure of identity information is punishable with imprisonment up to 3 years and a fine of up to Rs. 10,000 for individuals, or Rs. 1 lakh for companies. Using biometric information for any purpose other than Aadhaar is punishable with imprisonment up to 3 years and a fine of Rs. 10 lakh.",
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DocumentChunk:
    source: str
    chunk_idx: int
    text: str
    char_count: int


@dataclass
class ProcessingResult:
    docs_attempted: int
    docs_downloaded: int
    chunks_created: int
    train_samples: int
    valid_samples: int
    errors: List[str]


# ─────────────────────────────────────────────────────────────────────────────
# Download
# ─────────────────────────────────────────────────────────────────────────────

def download_pdf(url: str, dest: Path, timeout: int = 30) -> bool:
    """Download a PDF file; returns True on success."""
    if dest.exists():
        logger.info(f"[SKIP] Already downloaded: {dest.name}")
        return True
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36"
            )
        }
        resp = requests.get(url, headers=headers, timeout=timeout, stream=True)
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.success(f"[OK] Downloaded {dest.name} ({dest.stat().st_size // 1024} KB)")
        return True
    except Exception as e:
        logger.warning(f"[FAIL] Could not download {url}: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# PDF Parsing
# ─────────────────────────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract and clean text from a PDF using pdfplumber."""
    full_text = []
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page in pdf.pages:
                text = page.extract_text(x_tolerance=2, y_tolerance=3)
                if text:
                    full_text.append(text)
    except Exception as e:
        logger.error(f"[PDF ERROR] {pdf_path.name}: {e}")
        return ""

    raw = "\n".join(full_text)
    return _clean_text(raw)


def _clean_text(text: str) -> str:
    """Normalise whitespace and remove page artifacts."""
    # Remove repeated whitespace
    text = re.sub(r"[ \t]+", " ", text)
    # Remove excessive newlines (more than 2)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove form-feed characters
    text = text.replace("\f", "\n")
    # Remove header/footer noise (short lines < 5 chars)
    lines = [ln for ln in text.splitlines() if len(ln.strip()) > 4]
    return "\n".join(lines).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Chunking
# ─────────────────────────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    source: str,
    chunk_size: int = 512,
    overlap: int = 64,
) -> List[DocumentChunk]:
    """
    Word-based chunking with overlap.
    chunk_size and overlap are measured in words (approx. 0.75 tokens/word).
    """
    words = text.split()
    chunks: List[DocumentChunk] = []
    start = 0
    idx = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)

        if len(chunk_text) > 50:  # skip trivially short chunks
            chunks.append(
                DocumentChunk(
                    source=source,
                    chunk_idx=idx,
                    text=chunk_text,
                    char_count=len(chunk_text),
                )
            )
            idx += 1

        if end == len(words):
            break
        start = end - overlap  # slide window with overlap

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Instruction-Response Pair Generation
# ─────────────────────────────────────────────────────────────────────────────

INSTRUCTION_TEMPLATES = [
    "Explain the following section from UIDAI policy documents:\n\n{text}",
    "Summarize the key points from this UIDAI regulation excerpt:\n\n{text}",
    "What does the following UIDAI policy section say? Provide a clear explanation:\n\n{text}",
    "Based on the following UIDAI document excerpt, describe the obligations and provisions:\n\n{text}",
    "Extract the important guidelines from this UIDAI policy passage:\n\n{text}",
]


def format_as_gemma_chat(instruction: str, response: str) -> str:
    """
    Format a QA pair using Gemma's chat template.
    MLX-LM and HF both expect this format for instruction-tuned models.
    """
    return (
        "<start_of_turn>user\n"
        f"{instruction}"
        "<end_of_turn>\n"
        "<start_of_turn>model\n"
        f"{response}"
        "<end_of_turn>"
    )


def chunks_to_training_pairs(
    chunks: List[DocumentChunk],
) -> Generator[str, None, None]:
    """Convert text chunks into formatted Gemma training strings."""
    import random
    rng = random.Random(42)

    for chunk in chunks:
        template = rng.choice(INSTRUCTION_TEMPLATES)
        instruction = template.format(text=chunk.text[:800])  # cap instruction length

        # Response: a more structured version of the same chunk
        response = (
            f"According to UIDAI policy ({chunk.source}), this section covers:\n\n"
            f"{chunk.text}"
        )

        yield format_as_gemma_chat(instruction, response)

    # Also include static QA pairs
    for question, answer in STATIC_QA_PAIRS:
        yield format_as_gemma_chat(question, answer)


# ─────────────────────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_ingestion_pipeline(
    max_docs: int = 20,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    train_split: float = 0.8,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> ProcessingResult:
    """
    Full pipeline: download → parse → chunk → write JSONL.
    Returns a ProcessingResult with stats.
    """
    import random

    settings.ensure_dirs()
    result = ProcessingResult(
        docs_attempted=0, docs_downloaded=0,
        chunks_created=0, train_samples=0, valid_samples=0, errors=[]
    )

    docs_to_process = UIDAI_DOCUMENTS[:max_docs]
    all_chunks: List[DocumentChunk] = []

    def _log(msg: str) -> None:
        logger.info(msg)
        if progress_callback:
            progress_callback(msg)

    # Step 1: Download
    _log(f"Downloading {len(docs_to_process)} UIDAI documents...")
    for doc in tqdm(docs_to_process, desc="Downloading"):
        result.docs_attempted += 1
        dest = settings.raw_data_dir / doc["filename"]
        success = download_pdf(doc["url"], dest)
        if success:
            result.docs_downloaded += 1
        else:
            result.errors.append(f"Failed to download {doc['name']}: {doc['url']}")
        time.sleep(0.5)  # be polite to the server

    # Step 2: Parse & Chunk
    _log(f"Parsing {result.docs_downloaded} PDFs...")
    for doc in tqdm(docs_to_process, desc="Parsing"):
        pdf_path = settings.raw_data_dir / doc["filename"]
        if not pdf_path.exists():
            continue

        text = extract_text_from_pdf(pdf_path)
        if not text:
            result.errors.append(f"Empty text from {doc['filename']}")
            continue

        # Save processed text
        proc_path = settings.processed_data_dir / (pdf_path.stem + ".txt")
        proc_path.write_text(text, encoding="utf-8")

        # Chunk
        chunks = chunk_text(text, source=doc["name"], chunk_size=chunk_size, overlap=chunk_overlap)
        all_chunks.extend(chunks)
        result.chunks_created += len(chunks)
        _log(f"  → {doc['name']}: {len(chunks)} chunks")

    # Step 3: Generate training pairs & write JSONL
    _log("Building training dataset...")
    all_pairs = list(chunks_to_training_pairs(all_chunks))

    # Shuffle with seed
    rng = random.Random(42)
    rng.shuffle(all_pairs)

    split_idx = int(len(all_pairs) * train_split)
    train_pairs = all_pairs[:split_idx]
    valid_pairs = all_pairs[split_idx:]

    with open(settings.train_jsonl, "w", encoding="utf-8") as f:
        for text in train_pairs:
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

    with open(settings.valid_jsonl, "w", encoding="utf-8") as f:
        for text in valid_pairs:
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

    result.train_samples = len(train_pairs)
    result.valid_samples = len(valid_pairs)

    _log(
        f"Dataset ready: {result.train_samples} train / "
        f"{result.valid_samples} valid samples written."
    )
    return result
