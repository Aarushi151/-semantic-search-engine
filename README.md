# 🔍 Semantic Search Engine

AI-powered document retrieval using **FAISS**, **Sentence Transformers**, and **LangChain**.

## 📁 Project Structure

```
semantic-search-engine/
├── backend/
│   ├── main.py            ← FastAPI server (API endpoints)
│   ├── search_engine.py   ← Core logic (FAISS + embeddings + chunking)
│   └── requirements.txt   ← Python dependencies
├── frontend/
│   └── index.html         ← Single-file UI (no build step needed)
├── demo_notebook.ipynb    ← Interactive learning notebook
└── README.md
```

## 👩‍💻 Team & Ownership

| Module | Owner |
|--------|-------|
| Project Setup, Upload, Processing, Embeddings | Urvi Jain |
| FAISS Integration, Semantic Search, Top-K, Visualization | Aarushi Sharma |
| Keyword vs Semantic, Frontend UI, API, Testing, Deployment | Bhawna |

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Run the backend

```bash
cd backend
python main.py
# Server starts at http://localhost:8000
# API docs at  http://localhost:8000/docs
```

### 3. Open the frontend

Just open `frontend/index.html` in your browser — no build step required.

### 4. Run the notebook (optional)

```bash
pip install jupyter matplotlib
jupyter notebook demo_notebook.ipynb
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Check server + indexed doc count |
| POST | `/upload` | Upload a `.txt` or `.pdf` file |
| POST | `/search` | Semantic or keyword search |
| POST | `/compare` | Side-by-side semantic vs keyword |
| GET | `/documents/list` | List all indexed documents |
| DELETE | `/documents` | Clear all documents |

### Example — search request

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "how do machines learn", "top_k": 5, "search_type": "semantic"}'
```

---

## 🧠 Concepts Covered

| Concept | What it means |
|---------|---------------|
| **Embeddings** | Convert text → numerical vectors that capture meaning |
| **Vector similarity** | Find vectors closest to the query vector (cosine similarity) |
| **FAISS** | Facebook AI Similarity Search — fast nearest-neighbour lookup |
| **Sentence Transformers** | Pre-trained model (`all-MiniLM-L6-v2`) to create embeddings |
| **LangChain text splitter** | Splits documents into overlapping chunks for better retrieval |
| **Semantic search** | Finds results by *meaning*, even without exact word matches |
| **Keyword search** | Finds results by *exact term frequency* — misses synonyms |

### Keyword vs Semantic — key difference

| Query | Keyword finds | Semantic finds |
|-------|---------------|----------------|
| "clean energy" | docs with "clean", "energy" | docs about renewables, solar, wind |
| "coding in python" | docs with "coding", "python" | docs about programming, pip, functions |

---

## 🛠 Technologies

- **FastAPI** — REST API backend
- **Sentence Transformers** — `all-MiniLM-L6-v2` embedding model
- **FAISS** — vector index for fast similarity search
- **LangChain** — text chunking with `RecursiveCharacterTextSplitter`
- **PyMuPDF** — PDF text extraction

## ☁️ Deployment

To deploy on a cloud server (e.g. AWS EC2, Google Cloud, Railway):

```bash
# Install + run with gunicorn
pip install gunicorn
gunicorn -w 2 -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:8000
```

For the frontend, host `index.html` on any static file host (Netlify, GitHub Pages, Vercel) and update the `API` constant in the `<script>` to point to your deployed backend URL.
