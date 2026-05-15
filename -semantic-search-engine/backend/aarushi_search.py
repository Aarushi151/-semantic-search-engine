"""
=====================================================
  Aarushi Sharma — Semantic Search Module
=====================================================
  User Stories:
  1. FAISS Integration        — store & retrieve embeddings
  2. Semantic Search          — search by meaning
  3. Top-K Results            — ranking & retrieval logic
  4. Similarity Score Visualization
=====================================================
"""

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import re
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ─────────────────────────────────────────────
# USER STORY 1: FAISS Integration
# Sub-task: FAISS integration
# Acceptance: Embeddings stored and retrieved correctly
# ─────────────────────────────────────────────

class FAISSStore:
    """
    Manages a FAISS vector index for storing and retrieving embeddings.
    Uses IndexFlatL2 (exact L2 distance search) on normalized vectors,
    which is equivalent to cosine similarity search.
    """

    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        # Flat L2 index — exact search, no approximation
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.chunks: List[str] = []
        self.sources: List[str] = []

    def add_embeddings(self, embeddings: np.ndarray, chunks: List[str], source: str):
        """
        Store embeddings in FAISS index along with their text and source metadata.
        Embeddings must be float32 and normalized (unit vectors) for cosine similarity.
        """
        assert embeddings.dtype == np.float32, "Embeddings must be float32"
        assert embeddings.shape[1] == self.embedding_dim, "Embedding dimension mismatch"

        self.index.add(embeddings)
        self.chunks.extend(chunks)
        self.sources.extend([source] * len(chunks))
        print(f"[FAISS] Added {len(chunks)} vectors. Total: {self.index.ntotal}")

    def search(self, query_embedding: np.ndarray, top_k: int):
        """
        Search FAISS index for the top_k nearest vectors to the query embedding.
        Returns (distances, indices).
        """
        k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(query_embedding, k)
        return distances, indices

    def total(self) -> int:
        return self.index.ntotal

    def reset(self):
        self.index.reset()
        self.chunks.clear()
        self.sources.clear()


# ─────────────────────────────────────────────
# USER STORY 2: Semantic Search
# Sub-task: Search implementation
# Acceptance: Relevant semantic results returned
# ─────────────────────────────────────────────

class SemanticSearch:
    """
    Performs semantic search using Sentence Transformers + FAISS.
    Finds results by meaning, not exact keyword matching.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"[SemanticSearch] Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.store = FAISSStore(self.embedding_dim)

    def index_chunks(self, chunks: List[str], source: str):
        """Encode text chunks and store in FAISS."""
        print(f"[SemanticSearch] Encoding {len(chunks)} chunks from '{source}'...")
        embeddings = self.model.encode(
            chunks,
            normalize_embeddings=True,   # unit vectors → cosine similarity via L2
            show_progress_bar=True
        )
        embeddings = np.array(embeddings, dtype=np.float32)
        self.store.add_embeddings(embeddings, chunks, source)

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for semantically similar chunks.
        Returns ranked list with cosine similarity scores.
        """
        if self.store.total() == 0:
            raise ValueError("No documents indexed. Call index_chunks() first.")

        # Encode query the same way as documents
        query_vec = self.model.encode([query], normalize_embeddings=True)
        query_vec = np.array(query_vec, dtype=np.float32)

        distances, indices = self.store.search(query_vec, top_k)

        results = []
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:
                continue
            # For normalized vectors: cosine_similarity = 1 - (L2_distance² / 2)
            cosine_sim = float(1 - dist / 2)
            results.append({
                "rank": rank + 1,
                "text": self.store.chunks[idx],
                "source": self.store.sources[idx],
                "score": round(cosine_sim, 4),
            })

        return results


# ─────────────────────────────────────────────
# USER STORY 3: Top-K Results
# Sub-task: Ranking and retrieval logic
# Acceptance: Top-k results shown correctly
# ─────────────────────────────────────────────

def get_top_k_results(results: List[Dict], k: int) -> List[Dict]:
    """
    Return only the top-k results, sorted by score descending.
    Re-ranks if needed (FAISS already returns sorted, but useful for post-filtering).
    """
    sorted_results = sorted(results, key=lambda r: r["score"], reverse=True)
    top_k = sorted_results[:k]
    # Re-assign ranks after filtering
    for i, r in enumerate(top_k):
        r["rank"] = i + 1
    return top_k


def print_top_k(results: List[Dict], query: str):
    """Pretty-print top-K results to console."""
    print(f"\n{'='*60}")
    print(f"  Query: \"{query}\"")
    print(f"  Top-{len(results)} Results")
    print(f"{'='*60}")
    for r in results:
        bar = "█" * int(r["score"] * 20) + "░" * (20 - int(r["score"] * 20))
        print(f"\n  [{r['rank']}] {r['source']}")
        print(f"  Score: {r['score']:.4f}  {bar}")
        print(f"  Text : {r['text'][:120]}...")
    print()


# ─────────────────────────────────────────────
# USER STORY 4: Similarity Score Visualization
# Sub-task: Visualization implementation
# Acceptance: Scores visualized correctly
# ─────────────────────────────────────────────

def visualize_scores(semantic_results: List[Dict], keyword_results: List[Dict], query: str, save_path: str = "similarity_scores.png"):
    """
    Side-by-side bar chart comparing semantic vs keyword search scores.
    Saves chart as PNG.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#0f1117")
    fig.suptitle(f'Similarity Score Visualization\nQuery: "{query}"',
                 color="white", fontsize=13, fontweight="bold", y=1.01)

    def _plot(ax, results, title, bar_color, score_color):
        ax.set_facecolor("#1a1d27")
        if not results:
            ax.text(0.5, 0.5, "No results", ha="center", va="center", color="gray")
            return

        labels = [f"#{r['rank']}  {r['source']}\n{r['text'][:45]}..." for r in results]
        scores = [r["score"] for r in results]
        y_pos = range(len(results))

        bars = ax.barh(y_pos, scores, color=bar_color, alpha=0.85, edgecolor="none", height=0.6)

        # Score labels on bars
        for bar, score in zip(bars, scores):
            ax.text(
                bar.get_width() + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{score:.3f}",
                va="center", ha="left",
                color=score_color, fontsize=9, fontweight="bold"
            )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8, color="#c0c8d8")
        ax.set_xlim(0, 1.18)
        ax.set_xlabel("Cosine Similarity Score", color="#8892a4", fontsize=10)
        ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=10)
        ax.invert_yaxis()
        ax.tick_params(colors="#8892a4")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#2e3250")
        ax.spines["bottom"].set_color("#2e3250")
        ax.set_axisbelow(True)
        ax.xaxis.grid(True, color="#2e3250", linestyle="--", linewidth=0.5)

    _plot(axes[0], semantic_results, "🧠 Semantic Search", "#6366f1", "#a5b4fc")
    _plot(axes[1], keyword_results,  "🔑 Keyword Search",  "#f59e0b", "#fcd34d")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    print(f"[Visualization] Chart saved → {save_path}")


def visualize_score_heatmap(results: List[Dict], query: str, save_path: str = "score_heatmap.png"):
    """
    Heatmap showing similarity score per result chunk.
    """
    if not results:
        print("No results to visualize.")
        return

    scores = np.array([[r["score"] for r in results]])
    labels = [f"#{r['rank']} {r['source']}" for r in results]

    fig, ax = plt.subplots(figsize=(max(6, len(results) * 1.5), 2.5))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1d27")

    hm = ax.imshow(scores, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8, color="#c0c8d8")
    ax.set_yticks([])
    ax.set_title(f'Score Heatmap — Query: "{query}"', color="white", fontsize=11, pad=10)

    # Score text inside cells
    for j, r in enumerate(results):
        ax.text(j, 0, f"{r['score']:.3f}", ha="center", va="center",
                color="black", fontsize=9, fontweight="bold")

    cbar = plt.colorbar(hm, ax=ax, orientation="horizontal", pad=0.3, fraction=0.05)
    cbar.ax.tick_params(colors="#8892a4")
    cbar.set_label("Similarity Score", color="#8892a4")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    print(f"[Visualization] Heatmap saved → {save_path}")


# ─────────────────────────────────────────────
# DEMO — run this file directly to test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.append(".")

    # Sample documents
    docs = [
        ("ai.txt", [
            "Machine learning enables computers to learn patterns from data automatically.",
            "Deep learning uses multi-layered neural networks to process complex data.",
            "Natural language processing helps machines understand human text and speech.",
            "Reinforcement learning trains agents by rewarding correct decisions.",
        ]),
        ("climate.txt", [
            "Climate change is driven by rising greenhouse gas emissions from fossil fuels.",
            "Solar and wind energy are clean renewable alternatives to coal and gas.",
            "Deforestation reduces the Earth's ability to absorb carbon dioxide.",
            "The Paris Agreement targets limiting global warming to 1.5 degrees Celsius.",
        ]),
        ("python.txt", [
            "Python is a high-level interpreted language known for readable syntax.",
            "pip is the package manager used to install Python libraries.",
            "Functions in Python are defined with the def keyword.",
            "Virtual environments isolate project dependencies in Python.",
        ]),
    ]

    # --- Story 1 + 2: Index and search ---
    engine = SemanticSearch(model_name="all-MiniLM-L6-v2")

    for source, chunks in docs:
        engine.index_chunks(chunks, source)

    query = "how do computers learn from data"
    sem_results = engine.search(query, top_k=5)

    # --- Story 3: Top-K ---
    top3 = get_top_k_results(sem_results, k=3)
    print_top_k(top3, query)

    # --- Keyword search for comparison ---
    def keyword_search(chunks_all, sources_all, query, top_k=5):
        terms = set(re.findall(r'\w+', query.lower()))
        scored = []
        for i, chunk in enumerate(chunks_all):
            words = re.findall(r'\w+', chunk.lower())
            score = sum(chunk.lower().count(t) / (len(words) or 1) for t in terms)
            if score > 0:
                scored.append((score, i))
        scored.sort(reverse=True)
        return [{"rank": r+1, "text": chunks_all[i], "source": sources_all[i],
                 "score": round(min(s * 10, 1.0), 4)}
                for r, (s, i) in enumerate(scored[:top_k])]

    all_chunks = engine.store.chunks
    all_sources = engine.store.sources
    kw_results = keyword_search(all_chunks, all_sources, query, top_k=5)

    # --- Story 4: Visualize ---
    visualize_scores(sem_results, kw_results, query, save_path="similarity_scores.png")
    visualize_score_heatmap(sem_results, query, save_path="score_heatmap.png")
