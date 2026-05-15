"""
=====================================================
  Urvi Jain — Document Pipeline Module
=====================================================
  User Stories:
  1. Project Setup         — install dependencies
  2. Upload Documents      — file upload implementation
  3. Document Processing   — text preprocessing & chunking
  4. Embedding Generation  — embedding pipeline creation
=====================================================
"""

import os
import io
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

try:
    import fitz  # PyMuPDF
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    print("[Warning] PyMuPDF not installed. PDF support disabled. Run: pip install pymupdf")


# ─────────────────────────────────────────────
# USER STORY 1: Project Setup
# Sub-task: Install FAISS, LangChain, Sentence Transformers
# Acceptance: Project setup completed successfully
# ─────────────────────────────────────────────

REQUIRED_PACKAGES = [
    "faiss-cpu",
    "sentence-transformers",
    "langchain",
    "langchain-text-splitters",
    "pymupdf",
    "fastapi",
    "uvicorn",
    "python-multipart",
    "numpy",
]

def verify_setup() -> Dict[str, bool]:
    """
    Verify all required packages are installed.
    Run this first to confirm the project setup is complete.
    """
    import importlib
    status = {}
    package_map = {
        "faiss-cpu": "faiss",
        "sentence-transformers": "sentence_transformers",
        "langchain": "langchain",
        "langchain-text-splitters": "langchain_text_splitters",
        "pymupdf": "fitz",
        "fastapi": "fastapi",
        "uvicorn": "uvicorn",
        "python-multipart": "multipart",
        "numpy": "numpy",
    }
    for pkg, module in package_map.items():
        try:
            importlib.import_module(module)
            status[pkg] = True
        except ImportError:
            status[pkg] = False

    print("\n=== Project Setup Status ===")
    all_ok = True
    for pkg, ok in status.items():
        icon = "✅" if ok else "❌"
        print(f"  {icon} {pkg}")
        if not ok:
            all_ok = False
    print()
    if all_ok:
        print("✅ All dependencies installed. Project setup complete!\n")
    else:
        missing = [p for p, ok in status.items() if not ok]
        print(f"❌ Missing packages: {missing}")
        print(f"   Run: pip install {' '.join(missing)}\n")
    return status


# ─────────────────────────────────────────────
# USER STORY 2: Upload Documents
# Sub-task: File upload implementation
# Acceptance: Documents uploaded successfully
# ─────────────────────────────────────────────

SUPPORTED_TYPES = {
    "text/plain": [".txt"],
    "application/pdf": [".pdf"],
}

class DocumentUploader:
    """
    Handles document upload, validation, and storage.
    Supports .txt and .pdf files.
    """

    def __init__(self, upload_dir: str = "./uploaded_docs"):
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.uploaded_files: List[Dict] = []

    def validate_file(self, filename: str, content_type: str, size_bytes: int) -> Tuple[bool, str]:
        """Validate file type and size before accepting upload."""
        max_size = 10 * 1024 * 1024  # 10 MB limit

        if size_bytes > max_size:
            return False, f"File too large ({size_bytes/1024/1024:.1f} MB). Max 10 MB."

        ext = Path(filename).suffix.lower()
        allowed_extensions = []
        for exts in SUPPORTED_TYPES.values():
            allowed_extensions.extend(exts)

        if ext not in allowed_extensions:
            return False, f"Unsupported file type '{ext}'. Allowed: {allowed_extensions}"

        return True, "Valid"

    def save_file(self, content: bytes, filename: str) -> Dict:
        """
        Save uploaded file to disk and return metadata.
        Uses MD5 hash to detect duplicate uploads.
        """
        file_hash = hashlib.md5(content).hexdigest()
        save_path = self.upload_dir / filename

        # Check for duplicate
        for existing in self.uploaded_files:
            if existing["hash"] == file_hash:
                return {"status": "duplicate", "message": f"File already uploaded as '{existing['filename']}'", **existing}

        # Save to disk
        with open(save_path, "wb") as f:
            f.write(content)

        metadata = {
            "status": "success",
            "filename": filename,
            "path": str(save_path),
            "size_bytes": len(content),
            "hash": file_hash,
            "extension": Path(filename).suffix.lower(),
        }
        self.uploaded_files.append(metadata)
        print(f"[Upload] Saved '{filename}' ({len(content)/1024:.1f} KB)")
        return metadata

    def upload_from_path(self, file_path: str) -> Dict:
        """Upload a file from a local filesystem path (for testing)."""
        path = Path(file_path)
        if not path.exists():
            return {"status": "error", "message": f"File not found: {file_path}"}

        with open(path, "rb") as f:
            content = f.read()

        ext = path.suffix.lower()
        content_type = next((ct for ct, exts in SUPPORTED_TYPES.items() if ext in exts), "text/plain")
        valid, msg = self.validate_file(path.name, content_type, len(content))
        if not valid:
            return {"status": "error", "message": msg}

        return self.save_file(content, path.name)

    def list_uploads(self) -> List[Dict]:
        return self.uploaded_files


# ─────────────────────────────────────────────
# USER STORY 3: Document Processing
# Sub-task: Text preprocessing and chunking
# Acceptance: Documents processed correctly
# ─────────────────────────────────────────────

class DocumentProcessor:
    """
    Cleans and splits uploaded documents into chunks suitable for embedding.
    Uses LangChain's RecursiveCharacterTextSplitter.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""],
        )

    def extract_text(self, content: bytes, filename: str) -> str:
        """Extract raw text from file bytes (.txt or .pdf)."""
        ext = Path(filename).suffix.lower()

        if ext == ".pdf":
            if not PDF_SUPPORT:
                raise RuntimeError("PDF support requires PyMuPDF: pip install pymupdf")
            doc = fitz.open(stream=content, filetype="pdf")
            pages = [page.get_text() for page in doc]
            raw = "\n\n".join(pages)
            print(f"[Processor] Extracted {len(pages)} pages from PDF '{filename}'")
        else:
            try:
                raw = content.decode("utf-8")
            except UnicodeDecodeError:
                raw = content.decode("latin-1")
            print(f"[Processor] Loaded text file '{filename}' ({len(raw)} chars)")

        return raw

    def clean_text(self, text: str) -> str:
        """
        Clean raw text:
        - Remove excessive whitespace
        - Normalize line endings
        - Remove null bytes
        """
        text = text.replace("\x00", "")           # remove null bytes
        text = text.replace("\r\n", "\n")          # normalize line endings
        text = text.replace("\r", "\n")
        # collapse 3+ blank lines into 2
        import re
        text = re.sub(r"\n{3,}", "\n\n", text)
        # collapse multiple spaces (but not newlines)
        text = re.sub(r"[ \t]{2,}", " ", text)
        text = text.strip()
        return text

    def chunk_text(self, text: str) -> List[str]:
        """Split cleaned text into overlapping chunks using LangChain."""
        chunks = self.splitter.split_text(text)
        # Filter out chunks that are too short to be meaningful
        chunks = [c.strip() for c in chunks if len(c.strip()) > 30]
        return chunks

    def process(self, content: bytes, filename: str) -> Dict:
        """
        Full processing pipeline for one document:
        extract → clean → chunk.
        Returns dict with chunks and stats.
        """
        raw_text = self.extract_text(content, filename)
        clean = self.clean_text(raw_text)
        chunks = self.chunk_text(clean)

        stats = {
            "filename": filename,
            "raw_length": len(raw_text),
            "clean_length": len(clean),
            "num_chunks": len(chunks),
            "avg_chunk_len": int(sum(len(c) for c in chunks) / len(chunks)) if chunks else 0,
            "chunks": chunks,
        }

        print(f"[Processor] '{filename}': {len(chunks)} chunks "
              f"(avg {stats['avg_chunk_len']} chars, overlap={self.chunk_overlap})")
        return stats


# ─────────────────────────────────────────────
# USER STORY 4: Embedding Generation
# Sub-task: Embedding pipeline creation
# Acceptance: Embeddings generated successfully
# ─────────────────────────────────────────────

class EmbeddingPipeline:
    """
    Generates sentence embeddings using Sentence Transformers.
    Outputs normalized float32 vectors ready for FAISS indexing.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"[Embeddings] Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"[Embeddings] Model ready. Dimension: {self.embedding_dim}")

    def embed(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
        """
        Encode a list of texts into normalized embeddings.
        Returns float32 numpy array of shape (N, embedding_dim).
        normalize_embeddings=True → unit vectors → cosine similarity via L2.
        """
        if not texts:
            raise ValueError("Cannot embed empty list of texts.")

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        embeddings = embeddings.astype(np.float32)

        print(f"[Embeddings] Generated {len(embeddings)} embeddings, shape={embeddings.shape}")
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string. Returns shape (1, dim) float32 array."""
        vec = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
        return vec.astype(np.float32)

    def get_info(self) -> Dict:
        return {
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
        }


# ─────────────────────────────────────────────
# Full pipeline: Upload → Process → Embed
# ─────────────────────────────────────────────

class DocumentPipeline:
    """
    Combines Urvi's 4 modules into one end-to-end pipeline.
    Usage: pipeline.run(file_path) → embeddings + chunks
    """

    def __init__(self):
        self.uploader = DocumentUploader()
        self.processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)
        self.embedder = EmbeddingPipeline(model_name="all-MiniLM-L6-v2")

    def run(self, file_path: str) -> Dict:
        """Run the full pipeline on a local file."""
        print(f"\n{'─'*50}")
        print(f"  Pipeline: {file_path}")
        print(f"{'─'*50}")

        # Step 1: Upload
        upload_result = self.uploader.upload_from_path(file_path)
        if upload_result["status"] == "error":
            raise RuntimeError(f"Upload failed: {upload_result['message']}")

        # Step 2: Read content
        with open(upload_result["path"], "rb") as f:
            content = f.read()

        # Step 3: Process
        proc_result = self.processor.process(content, upload_result["filename"])
        chunks = proc_result["chunks"]
        if not chunks:
            raise ValueError("No chunks produced from document.")

        # Step 4: Embed
        embeddings = self.embedder.embed(chunks, show_progress=False)

        return {
            "filename": upload_result["filename"],
            "num_chunks": len(chunks),
            "embedding_dim": embeddings.shape[1],
            "embeddings": embeddings,
            "chunks": chunks,
        }


# ─────────────────────────────────────────────
# DEMO — run directly to test Urvi's pipeline
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Step 1: Verify setup
    verify_setup()

    # Step 2: Create a sample .txt file for demo
    sample_path = "sample_urvi.txt"
    with open(sample_path, "w") as f:
        f.write("""Artificial Intelligence and Machine Learning

Artificial intelligence (AI) refers to the simulation of human intelligence in machines.
These machines are programmed to think like humans and mimic their actions.

Machine learning is a subset of AI that provides systems the ability to automatically
learn and improve from experience without being explicitly programmed.

Deep learning is part of a broader family of machine learning methods based on
artificial neural networks with representation learning.

Natural language processing (NLP) is a subfield of linguistics, computer science,
and artificial intelligence concerned with the interactions between computers and
human language.

Computer vision is an interdisciplinary scientific field that deals with how computers
can gain high-level understanding from digital images or videos.""")

    print(f"Created sample file: {sample_path}\n")

    # Step 3: Run the pipeline
    pipeline = DocumentPipeline()
    result = pipeline.run(sample_path)

    print(f"\n✅ Pipeline complete!")
    print(f"   File      : {result['filename']}")
    print(f"   Chunks    : {result['num_chunks']}")
    print(f"   Emb shape : {result['embeddings'].shape}")
    print(f"\n   First chunk preview:")
    print(f"   \"{result['chunks'][0][:120]}...\"")
    print(f"\n   Embedding[0] first 5 values: {result['embeddings'][0][:5]}")

    # Cleanup
    os.remove(sample_path)
