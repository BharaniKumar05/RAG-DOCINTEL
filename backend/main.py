"""
FastAPI Server for AI-Powered Document Intelligence & RAG System.
Handles document ingestion, dual-index management, SSE streaming RAG chat,
vector projections, and system configuration.
"""

import os
import sys
import json
import uuid
import asyncio
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# Ensure workspace root is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from backend.document_parser import DocumentParser
from backend.chunking import DocumentChunker, Chunk
from backend.retrieval_engine import HybridRetrievalEngine
from backend.rag_pipeline import RAGPipeline
from backend.sample_corpus import SAMPLE_DOCUMENTS

app = FastAPI(
    title="AI Document Intelligence & RAG System",
    description="Production-grade hybrid RAG engine with multi-format parsing, 2D vector space visualizer, and live execution tracing.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State
documents_db: Dict[str, Dict[str, Any]] = {}
all_chunks: List[Chunk] = []

chunk_config = {
    "strategy": "semantic",
    "chunk_size": 600,
    "chunk_overlap": 120
}

retrieval_engine = HybridRetrievalEngine(rrf_k=60, dense_weight=0.5, bm25_weight=0.5)
rag_pipeline = RAGPipeline(retrieval_engine)


def rebuild_index():
    """Rebuild dual vector & BM25 index from all documents."""
    global all_chunks
    all_chunks = []
    
    for doc_id, doc in documents_db.items():
        doc_chunks = DocumentChunker.chunk_document(
            doc_id=doc_id,
            doc_title=doc["metadata"]["title"],
            text=doc["text"],
            strategy=chunk_config["strategy"],
            chunk_size=chunk_config["chunk_size"],
            chunk_overlap=chunk_config["chunk_overlap"]
        )
        doc["chunks"] = [c.to_dict() for c in doc_chunks]
        all_chunks.extend(doc_chunks)

    retrieval_engine.index_chunks(all_chunks)


def load_initial_corpus():
    """Load default domain sample documents."""
    documents_db.clear()
    for sample in SAMPLE_DOCUMENTS:
        doc_id = f"doc_{uuid.uuid4().hex[:8]}"
        parsed = DocumentParser.parse_text(
            filename=sample["filename"],
            text=sample["content"],
            doc_type="markdown"
        )
        # Override title with sample title
        parsed["metadata"]["title"] = sample["title"]
        parsed["doc_id"] = doc_id
        documents_db[doc_id] = parsed

    rebuild_index()


# Initialize sample documents on startup
load_initial_corpus()


# Request Models
class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 4
    temperature: Optional[float] = 0.2
    doc_filter: Optional[str] = None


class ConfigRequest(BaseModel):
    provider: str
    api_key: Optional[str] = ""
    model_name: Optional[str] = ""
    chunk_strategy: Optional[str] = "semantic"
    chunk_size: Optional[int] = 600
    chunk_overlap: Optional[int] = 120


# API Endpoints
@app.get("/api/health")
async def health_check():
    return {
        "status": "online",
        "total_documents": len(documents_db),
        "total_chunks": len(all_chunks),
        "active_provider": rag_pipeline.active_provider,
        "model": rag_pipeline.model_name
    }


@app.get("/api/documents")
async def list_documents():
    """Return list of all documents with their metadata and entity summaries."""
    doc_list = []
    for doc_id, doc in documents_db.items():
        meta = dict(doc["metadata"])
        doc_list.append({
            "doc_id": doc_id,
            "filename": doc["filename"],
            "title": meta.get("title", doc["filename"]),
            "doc_type": meta.get("doc_type", "text"),
            "word_count": meta.get("word_count", 0),
            "char_count": meta.get("char_count", 0),
            "reading_time_minutes": meta.get("reading_time_minutes", 1),
            "summary": meta.get("summary", ""),
            "entities": meta.get("entities", {}),
            "keywords": meta.get("keywords", []),
            "sections": meta.get("sections", []),
            "chunk_count": len(doc.get("chunks", []))
        })
    return {"documents": doc_list, "total_chunks": len(all_chunks)}


@app.get("/api/documents/{doc_id}")
async def get_document(doc_id: str):
    """Retrieve full text and chunks of a specific document."""
    if doc_id not in documents_db:
        raise HTTPException(status_code=404, detail="Document not found")
    doc = documents_db[doc_id]
    return {
        "doc_id": doc_id,
        "filename": doc["filename"],
        "text": doc["text"],
        "metadata": doc["metadata"],
        "chunks": doc.get("chunks", [])
    }


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document from database and reindex remaining."""
    if doc_id not in documents_db:
        raise HTTPException(status_code=404, detail="Document not found")
    del documents_db[doc_id]
    rebuild_index()
    return {"message": "Document deleted successfully", "remaining_docs": len(documents_db)}


@app.post("/api/documents/reset-samples")
async def reset_sample_documents():
    """Reset knowledge base to default 4 industry documents."""
    load_initial_corpus()
    return {"message": "Sample documents restored successfully", "total_docs": len(documents_db), "total_chunks": len(all_chunks)}


@app.post("/api/documents/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload and parse one or multiple documents (PDF, MD, TXT, CSV, JSON)."""
    uploaded_records = []
    
    for file in files:
        try:
            content = await file.read()
            parsed = DocumentParser.parse_bytes(filename=file.filename, content=content)
            doc_id = f"doc_{uuid.uuid4().hex[:8]}"
            parsed["doc_id"] = doc_id
            documents_db[doc_id] = parsed
            uploaded_records.append({
                "doc_id": doc_id,
                "filename": file.filename,
                "title": parsed["metadata"]["title"],
                "word_count": parsed["metadata"]["word_count"],
                "summary": parsed["metadata"]["summary"]
            })
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse {file.filename}: {str(e)}")

    rebuild_index()
    return {
        "message": f"Successfully ingested and indexed {len(uploaded_records)} document(s)",
        "documents": uploaded_records,
        "total_chunks": len(all_chunks)
    }


@app.get("/api/vector-space")
async def get_vector_space():
    """Return 2D coordinates for all indexed chunks for visual scatter plotting."""
    nodes = []
    for chunk in all_chunks:
        coords = retrieval_engine.coords_2d.get(chunk.chunk_id, (0.0, 0.0))
        nodes.append({
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "doc_title": chunk.doc_title,
            "section_title": chunk.section_title,
            "snippet": chunk.text[:120] + "..." if len(chunk.text) > 120 else chunk.text,
            "x": coords[0],
            "y": coords[1],
            "word_count": chunk.word_count
        })
    return {
        "nodes": nodes,
        "total_nodes": len(nodes),
        "total_documents": len(documents_db)
    }


@app.post("/api/system/config")
async def update_config(config: ConfigRequest):
    """Update AI provider and chunking configuration."""
    global chunk_config
    
    rag_pipeline.configure_provider(
        provider=config.provider,
        api_key=config.api_key or "",
        model_name=config.model_name or ""
    )
    
    strategy_changed = (
        config.chunk_strategy != chunk_config["strategy"] or
        config.chunk_size != chunk_config["chunk_size"] or
        config.chunk_overlap != chunk_config["chunk_overlap"]
    )
    
    if strategy_changed:
        chunk_config["strategy"] = config.chunk_strategy or "semantic"
        chunk_config["chunk_size"] = config.chunk_size or 600
        chunk_config["chunk_overlap"] = config.chunk_overlap or 120
        rebuild_index()

    return {
        "message": "Configuration updated successfully",
        "active_provider": rag_pipeline.active_provider,
        "model_name": rag_pipeline.model_name,
        "chunk_config": chunk_config,
        "total_chunks": len(all_chunks)
    }


@app.get("/api/system/stats")
async def get_system_stats():
    """Retrieve full telemetry, configuration, and index statistics."""
    total_words = sum(d["metadata"].get("word_count", 0) for d in documents_db.values())
    return {
        "documents_count": len(documents_db),
        "chunks_count": len(all_chunks),
        "total_words": total_words,
        "active_provider": rag_pipeline.active_provider,
        "active_model": rag_pipeline.model_name,
        "chunk_strategy": chunk_config["strategy"],
        "chunk_size": chunk_config["chunk_size"],
        "chunk_overlap": chunk_config["chunk_overlap"],
        "retriever": "Hybrid Dense (TF-IDF Cosine) + Sparse BM25 (Okapi) + RRF (k=60) + Cross-Encoder Reranker"
    }


@app.post("/api/rag/chat")
async def rag_chat_stream(request: QueryRequest):
    """Server-Sent Events (SSE) streaming endpoint for RAG queries."""
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    async def event_generator():
        try:
            async for event in rag_pipeline.execute_rag_stream(
                query=query,
                top_k=request.top_k or 4,
                temperature=request.temperature or 0.2,
                doc_filter=request.doc_filter
            ):
                payload = json.dumps(event)
                yield f"data: {payload}\n\n"
        except Exception as e:
            err_payload = json.dumps({"type": "error", "message": str(e)})
            yield f"data: {err_payload}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


# Serve frontend static assets
frontend_dir = os.path.join(parent_dir, "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

    @app.get("/")
    async def serve_index():
        return FileResponse(os.path.join(frontend_dir, "index.html"))
