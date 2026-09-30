"""
Runner script for AI-Powered Document Intelligence & RAG System.
Initializes knowledge base and starts FastAPI / Uvicorn server.
"""

import sys
import os
import uvicorn

# Add current workspace root to PYTHONPATH
workspace_dir = os.path.dirname(os.path.abspath(__file__))

if workspace_dir not in sys.path:
    sys.path.insert(0, workspace_dir)


if __name__ == "__main__":
    print("\n========================================================")
    print(" [AI Document Intelligence & Hybrid RAG System v2.0]")
    print("========================================================")
    print(" [*] Multi-Format Ingestion: PDF, TXT, MD, CSV, JSON")
    print(" [*] Dual Index: Sublinear Dense Vector + Okapi BM25")
    print(" [*] Hybrid Retrieval: Reciprocal Rank Fusion (RRF)")
    print(" [*] Context Reranker: Cross-Encoder calibrated scoring")
    print(" [*] 2D Vector Space Canvas: Live PCA Projections")
    print(" [*] Web UI & SSE Stream")
    print("========================================================\n")

    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=False,
        log_level="info"
    )