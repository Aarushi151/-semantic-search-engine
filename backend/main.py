from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import uvicorn

from search_engine import SemanticSearchEngine

app = FastAPI(title="Semantic Search Engine API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = SemanticSearchEngine()


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    search_type: str = "semantic"  # "semantic" or "keyword"


class SearchResult(BaseModel):
    text: str
    score: float
    source: str
    rank: int


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    search_type: str
    total_docs: int


@app.get("/")
def root():
    return {"message": "Semantic Search Engine API is running", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok", "documents_indexed": engine.get_doc_count()}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a text or PDF document and index it."""
    allowed_types = ["text/plain", "application/pdf"]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Only .txt and .pdf files are supported.")

    content = await file.read()

    try:
        num_chunks = engine.add_document(content, filename=file.filename, content_type=file.content_type)
        return {
            "message": f"Document '{file.filename}' uploaded and indexed successfully.",
            "chunks_created": num_chunks,
            "total_documents": engine.get_doc_count()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest):
    """Search documents using semantic or keyword search."""
    if not engine.get_doc_count():
        raise HTTPException(status_code=400, detail="No documents indexed yet. Please upload documents first.")

    if request.search_type == "semantic":
        results = engine.semantic_search(request.query, top_k=request.top_k)
    elif request.search_type == "keyword":
        results = engine.keyword_search(request.query, top_k=request.top_k)
    else:
        raise HTTPException(status_code=400, detail="search_type must be 'semantic' or 'keyword'")

    return SearchResponse(
        query=request.query,
        results=results,
        search_type=request.search_type,
        total_docs=engine.get_doc_count()
    )


@app.post("/compare")
def compare_search(request: SearchRequest):
    """Compare semantic vs keyword search results side by side."""
    if not engine.get_doc_count():
        raise HTTPException(status_code=400, detail="No documents indexed yet.")

    semantic = engine.semantic_search(request.query, top_k=request.top_k)
    keyword = engine.keyword_search(request.query, top_k=request.top_k)

    return {
        "query": request.query,
        "semantic_results": semantic,
        "keyword_results": keyword
    }


@app.delete("/documents")
def clear_documents():
    """Clear all indexed documents."""
    engine.clear()
    return {"message": "All documents cleared.", "total_documents": 0}


@app.get("/documents/list")
def list_documents():
    """List all indexed document sources."""
    return {"documents": engine.list_sources(), "total": engine.get_doc_count()}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
