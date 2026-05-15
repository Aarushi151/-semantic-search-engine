import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict, Optional
import re
import io

try:
    import fitz  # PyMuPDF for PDF parsing
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False


class SemanticSearchEngine:
    """
    Core semantic search engine using:
    - Sentence Transformers for embedding generation
    - FAISS for vector similarity search
    - LangChain for text splitting
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", chunk_size: int = 500, chunk_overlap: int = 50):
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        # FAISS index (L2 distance; we convert to cosine similarity)
        self.index = faiss.IndexFlatL2(self.embedding_dim)

        # Metadata store
        self.chunks: List[str] = []
        self.sources: List[str] = []

        # LangChain text splitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        print("Search engine ready.")

    def _extract_text(self, content: bytes, filename: str, content_type: str) -> str:
        """Extract plain text from uploaded file bytes."""
        if content_type == "application/pdf":
            if not PDF_SUPPORT:
                raise RuntimeError("PDF support requires PyMuPDF. Install with: pip install pymupdf")
            doc = fitz.open(stream=content, filetype="pdf")
            return "\n\n".join(page.get_text() for page in doc)
        else:
            # Plain text — try utf-8, fallback to latin-1
            try:
                return content.decode("utf-8")
            except UnicodeDecodeError:
                return content.decode("latin-1")

    def add_document(self, content: bytes, filename: str, content_type: str) -> int:
        """Process, embed and index a document. Returns number of chunks created."""
        raw_text = self._extract_text(content, filename, content_type)
        chunks = self.splitter.split_text(raw_text)

        if not chunks:
            raise ValueError("Document produced no text chunks after processing.")

        embeddings = self.model.encode(chunks, show_progress_bar=False, normalize_embeddings=True)
        embeddings = np.array(embeddings, dtype=np.float32)

        self.index.add(embeddings)
        self.chunks.extend(chunks)
        self.sources.extend([filename] * len(chunks))

        return len(chunks)

    def semantic_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search using vector similarity (semantic meaning)."""
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        query_embedding = np.array(query_embedding, dtype=np.float32)

        k = min(top_k, len(self.chunks))
        distances, indices = self.index.search(query_embedding, k)

        results = []
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:
                continue
            # Convert L2 distance of normalized vectors to cosine similarity
            cosine_sim = float(1 - dist / 2)
            results.append({
                "text": self.chunks[idx],
                "score": round(cosine_sim, 4),
                "source": self.sources[idx],
                "rank": rank + 1
            })

        return results

    def keyword_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search using simple TF-style keyword matching (BM25-lite)."""
        query_terms = set(re.findall(r'\w+', query.lower()))

        scored = []
        for i, chunk in enumerate(self.chunks):
            chunk_lower = chunk.lower()
            chunk_words = re.findall(r'\w+', chunk_lower)
            total_words = len(chunk_words) or 1

            # Term frequency score
            score = sum(chunk_lower.count(term) / total_words for term in query_terms)
            if score > 0:
                scored.append((score, i))

        scored.sort(reverse=True)
        top = scored[:top_k]

        results = []
        for rank, (score, idx) in enumerate(top):
            results.append({
                "text": self.chunks[idx],
                "score": round(min(score * 10, 1.0), 4),  # normalize to 0-1
                "source": self.sources[idx],
                "rank": rank + 1
            })

        return results

    def get_doc_count(self) -> int:
        return self.index.ntotal

    def list_sources(self) -> List[str]:
        return list(set(self.sources))

    def clear(self):
        self.index.reset()
        self.chunks.clear()
        self.sources.clear()
