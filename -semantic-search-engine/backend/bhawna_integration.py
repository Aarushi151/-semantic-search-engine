"""
=====================================================
  Bhawna — Integration, Testing & Deployment Module
=====================================================
  User Stories:
  1. Keyword vs Semantic Search  — comparison module
  2. Frontend UI                 — UI development notes + API connector
  3. API Integration             — FastAPI backend integration
  4. Testing & Validation        — testing and debugging
  5. Deployment                  — cloud deployment guide
=====================================================
"""

import re
import time
import json
import requests
from typing import List, Dict, Optional
from pathlib import Path


# ─────────────────────────────────────────────
# USER STORY 1: Keyword vs Semantic Search
# Sub-task: Search comparison module
# Acceptance: Comparison works correctly
# ─────────────────────────────────────────────

class SearchComparison:
    """
    Demonstrates and compares keyword search vs semantic search.
    Shows clearly where semantic search outperforms keyword matching.
    """

    @staticmethod
    def keyword_search(chunks: List[str], sources: List[str], query: str, top_k: int = 5) -> List[Dict]:
        """
        Keyword / TF search: counts term frequency in each chunk.
        Fails on synonyms and paraphrases — no understanding of meaning.
        """
        query_terms = set(re.findall(r'\w+', query.lower()))
        scored = []

        for i, chunk in enumerate(chunks):
            words = re.findall(r'\w+', chunk.lower())
            total = len(words) or 1
            score = sum(chunk.lower().count(t) / total for t in query_terms)
            if score > 0:
                scored.append((score, i))

        scored.sort(reverse=True)
        return [
            {
                "rank": r + 1,
                "text": chunks[i],
                "source": sources[i],
                "score": round(min(s * 10, 1.0), 4),
                "method": "keyword"
            }
            for r, (s, i) in enumerate(scored[:top_k])
        ]

    @staticmethod
    def compare(semantic_results: List[Dict], keyword_results: List[Dict], query: str) -> Dict:
        """
        Side-by-side comparison of semantic vs keyword results.
        Returns analysis of overlap and differences.
        """
        sem_texts  = {r["text"][:80] for r in semantic_results}
        key_texts  = {r["text"][:80] for r in keyword_results}
        overlap    = sem_texts & key_texts
        sem_only   = sem_texts - key_texts
        key_only   = key_texts - sem_texts

        print(f"\n{'='*60}")
        print(f"  Query: \"{query}\"")
        print(f"{'='*60}")
        print(f"\n  {'🧠 SEMANTIC':^28}  {'🔑 KEYWORD':^28}")
        print(f"  {'-'*28}  {'-'*28}")

        max_r = max(len(semantic_results), len(keyword_results))
        for i in range(max_r):
            sem = semantic_results[i] if i < len(semantic_results) else None
            key = keyword_results[i]  if i < len(keyword_results)  else None
            sem_str = f"[{sem['rank']}] {sem['text'][:22]}... ({sem['score']:.3f})" if sem else ""
            key_str = f"[{key['rank']}] {key['text'][:22]}... ({key['score']:.3f})" if key else ""
            print(f"  {sem_str:<28}  {key_str:<28}")

        print(f"\n  Overlap   : {len(overlap)} result(s) in common")
        print(f"  Sem-only  : {len(sem_only)} result(s) found only by semantic")
        print(f"  Key-only  : {len(key_only)} result(s) found only by keyword")

        return {
            "query": query,
            "semantic": semantic_results,
            "keyword": keyword_results,
            "overlap_count": len(overlap),
            "semantic_only_count": len(sem_only),
            "keyword_only_count": len(key_only),
        }

    @staticmethod
    def demo_difference():
        """
        Illustrates when semantic beats keyword with real examples.
        No backend needed — just explains the difference clearly.
        """
        print("\n🔍 KEYWORD vs SEMANTIC SEARCH — Key Differences\n")

        cases = [
            {
                "query": "how do computers learn from experience",
                "keyword_finds": "texts containing 'computers', 'learn', 'experience'",
                "semantic_finds": "texts about machine learning, training, model updates",
                "why_semantic_wins": "Understands meaning — 'machine learning' = 'computers learning'"
            },
            {
                "query": "clean energy sources",
                "keyword_finds": "texts containing 'clean', 'energy', 'sources'",
                "semantic_finds": "texts about solar, wind, renewables, sustainability",
                "why_semantic_wins": "Synonyms — 'clean energy' ≈ 'renewable energy' ≈ 'solar/wind'"
            },
            {
                "query": "installing Python packages",
                "keyword_finds": "texts containing 'installing', 'Python', 'packages'",
                "semantic_finds": "texts about pip, virtual environments, dependencies",
                "why_semantic_wins": "Related concepts — pip is how you install Python packages"
            },
        ]

        for case in cases:
            print(f"  Query      : \"{case['query']}\"")
            print(f"  Keyword  → {case['keyword_finds']}")
            print(f"  Semantic → {case['semantic_finds']}")
            print(f"  Why semantic wins: {case['why_semantic_wins']}\n")


# ─────────────────────────────────────────────
# USER STORY 2 & 3: Frontend UI + API Integration
# Sub-task: UI development + FastAPI backend integration
# Acceptance: APIs work, users can upload and search
# ─────────────────────────────────────────────

class APIClient:
    """
    Python client for the Semantic Search Engine API.
    Wraps all FastAPI endpoints for easy testing.
    """

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def health_check(self) -> Dict:
        """Check if the backend is running."""
        try:
            r = self.session.get(f"{self.base_url}/health", timeout=5)
            return r.json()
        except requests.ConnectionError:
            return {"status": "error", "message": "Cannot connect. Is the server running?"}

    def upload_file(self, file_path: str) -> Dict:
        """Upload a document file to the backend."""
        path = Path(file_path)
        if not path.exists():
            return {"error": f"File not found: {file_path}"}

        with open(path, "rb") as f:
            files = {"file": (path.name, f, "text/plain" if path.suffix == ".txt" else "application/pdf")}
            r = self.session.post(f"{self.base_url}/upload", files=files)

        return r.json()

    def upload_text(self, text: str, filename: str = "document.txt") -> Dict:
        """Upload raw text as a .txt file."""
        files = {"file": (filename, text.encode(), "text/plain")}
        r = self.session.post(f"{self.base_url}/upload", files=files)
        return r.json()

    def search(self, query: str, top_k: int = 5, search_type: str = "semantic") -> Dict:
        """Run a search query."""
        payload = {"query": query, "top_k": top_k, "search_type": search_type}
        r = self.session.post(f"{self.base_url}/search", json=payload)
        return r.json()

    def compare(self, query: str, top_k: int = 5) -> Dict:
        """Run both semantic and keyword search and compare."""
        payload = {"query": query, "top_k": top_k}
        r = self.session.post(f"{self.base_url}/compare", json=payload)
        return r.json()

    def list_documents(self) -> Dict:
        """List all indexed documents."""
        r = self.session.get(f"{self.base_url}/documents/list")
        return r.json()

    def clear_documents(self) -> Dict:
        """Clear all indexed documents."""
        r = self.session.delete(f"{self.base_url}/documents")
        return r.json()

    def print_results(self, results: List[Dict]):
        """Pretty-print search results."""
        for r in results:
            bar = "█" * int(r["score"] * 15) + "░" * (15 - int(r["score"] * 15))
            print(f"\n  [{r['rank']}] {r['source']} | {bar} {r['score']:.3f}")
            print(f"      {r['text'][:120]}...")


# ─────────────────────────────────────────────
# USER STORY 4: Testing & Validation
# Sub-task: Testing and debugging
# Acceptance: System works without issues
# ─────────────────────────────────────────────

class TestSuite:
    """
    End-to-end tests for the Semantic Search Engine.
    Validates all API endpoints and search quality.
    """

    def __init__(self, api_url: str = "http://localhost:8000"):
        self.client = APIClient(api_url)
        self.passed = 0
        self.failed = 0
        self.errors = []

    def _assert(self, condition: bool, test_name: str, detail: str = ""):
        if condition:
            print(f"  ✅ PASS — {test_name}")
            self.passed += 1
        else:
            print(f"  ❌ FAIL — {test_name}: {detail}")
            self.failed += 1
            self.errors.append(test_name)

    def test_health(self):
        print("\n[Test] Health check")
        result = self.client.health_check()
        self._assert(result.get("status") == "ok", "Server is running")

    def test_upload(self):
        print("\n[Test] Document upload")
        result = self.client.upload_text(
            "Machine learning is a subset of AI. Python is widely used for data science.",
            filename="test_upload.txt"
        )
        self._assert("chunks_created" in result, "Upload returns chunk count")
        self._assert(result.get("chunks_created", 0) > 0, "At least 1 chunk created")

    def test_semantic_search(self):
        print("\n[Test] Semantic search")
        result = self.client.search("how do machines learn", top_k=3, search_type="semantic")
        self._assert("results" in result, "Semantic search returns results key")
        self._assert(len(result.get("results", [])) > 0, "At least 1 result returned")
        self._assert(result.get("search_type") == "semantic", "Search type is semantic")

    def test_keyword_search(self):
        print("\n[Test] Keyword search")
        result = self.client.search("machine learning AI Python", top_k=3, search_type="keyword")
        self._assert("results" in result, "Keyword search returns results key")
        self._assert(len(result.get("results", [])) > 0, "At least 1 keyword result returned")

    def test_compare(self):
        print("\n[Test] Compare endpoint")
        result = self.client.compare("data science", top_k=3)
        self._assert("semantic_results" in result, "Compare has semantic_results")
        self._assert("keyword_results" in result, "Compare has keyword_results")

    def test_scores_in_range(self):
        print("\n[Test] Score validation")
        result = self.client.search("machine learning", top_k=5)
        scores = [r["score"] for r in result.get("results", [])]
        valid = all(0.0 <= s <= 1.0 for s in scores)
        self._assert(valid, f"All scores in [0,1] range: {scores}")

    def test_ranking_order(self):
        print("\n[Test] Ranking order")
        result = self.client.search("machine learning", top_k=5)
        results = result.get("results", [])
        if len(results) > 1:
            scores = [r["score"] for r in results]
            self._assert(scores == sorted(scores, reverse=True), "Results sorted by score descending")
        else:
            print("  ⚠️  SKIP — not enough results to check order")

    def test_empty_query_handling(self):
        print("\n[Test] Edge cases")
        # Empty list after clearing
        self.client.clear_documents()
        result = self.client.search("test", top_k=3)
        self._assert("detail" in result or result.get("results") == [], "Empty index handled gracefully")

    def run_all(self) -> bool:
        print("\n" + "="*60)
        print("  SEMANTIC SEARCH ENGINE — TEST SUITE")
        print("="*60)

        start = time.time()
        self.test_health()
        self.test_upload()
        self.test_semantic_search()
        self.test_keyword_search()
        self.test_compare()
        self.test_scores_in_range()
        self.test_ranking_order()
        self.test_empty_query_handling()

        elapsed = time.time() - start
        print(f"\n{'='*60}")
        print(f"  Results: {self.passed} passed, {self.failed} failed  ({elapsed:.2f}s)")
        if self.errors:
            print(f"  Failed tests: {self.errors}")
        print("="*60 + "\n")
        return self.failed == 0


# ─────────────────────────────────────────────
# USER STORY 5: Deployment
# Sub-task: Cloud deployment
# Acceptance: Application accessible successfully
# ─────────────────────────────────────────────

DEPLOYMENT_GUIDE = """
=============================================================
  DEPLOYMENT GUIDE — Semantic Search Engine
=============================================================

OPTION 1: Local (development)
──────────────────────────────
  cd backend
  pip install -r requirements.txt
  python main.py
  # → http://localhost:8000
  # Open frontend/index.html in browser

OPTION 2: Railway (recommended for quick cloud deploy)
──────────────────────────────────────────────────────
  1. Push project to GitHub
  2. Go to https://railway.app → New Project → Deploy from GitHub
  3. Set start command: uvicorn main:app --host 0.0.0.0 --port $PORT
  4. Railway auto-assigns a public URL
  5. Update API constant in frontend/index.html to that URL

OPTION 3: AWS EC2 / Google Cloud VM
──────────────────────────────────────────────────────
  # On server:
  git clone <your-repo>
  cd semantic-search-engine/backend
  pip install -r requirements.txt
  pip install gunicorn
  gunicorn -w 2 -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:8000 --daemon

  # Open port 8000 in security group / firewall
  # Access at http://<server-ip>:8000

OPTION 4: Docker
──────────────────────────────────────────────────────
  FROM python:3.11-slim
  WORKDIR /app
  COPY backend/ .
  RUN pip install -r requirements.txt
  EXPOSE 8000
  CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

  docker build -t semantic-search .
  docker run -p 8000:8000 semantic-search

Frontend hosting (any option):
  → Netlify: drag-drop frontend/ folder at netlify.com
  → GitHub Pages: push frontend/ and enable Pages in repo settings
  → Update API URL in index.html to your backend's public URL

=============================================================
"""


def print_deployment_guide():
    print(DEPLOYMENT_GUIDE)


# ─────────────────────────────────────────────
# DEMO — run directly
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Run test suite against live server
        suite = TestSuite()
        success = suite.run_all()
        sys.exit(0 if success else 1)

    elif len(sys.argv) > 1 and sys.argv[1] == "deploy":
        print_deployment_guide()

    else:
        # Demo: show search comparison difference (no server needed)
        SearchComparison.demo_difference()

        # Demo: API client usage (requires server running)
        print("\n─── API Client Demo (requires backend running) ───\n")
        client = APIClient()
        health = client.health_check()
        print(f"Server status: {health}\n")

        if health.get("status") == "ok":
            # Upload sample
            client.upload_text(
                "Machine learning allows computers to learn patterns from data. "
                "Python is the most popular language for AI and data science. "
                "Deep learning uses neural networks to solve complex problems.",
                filename="bhawna_demo.txt"
            )

            # Search
            print("Semantic search: 'how do machines understand data'")
            result = client.search("how do machines understand data", top_k=3)
            client.print_results(result.get("results", []))

            # Compare
            print("\nCompare: 'python programming'")
            compare = client.compare("python programming", top_k=3)
            print(f"Semantic: {len(compare.get('semantic_results', []))} results")
            print(f"Keyword : {len(compare.get('keyword_results', []))} results")
        else:
            print("Start the backend first: python main.py")
            print("\nThen run tests with: python bhawna_integration.py test")
            print("Or see deploy options: python bhawna_integration.py deploy")
