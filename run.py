"""
Runner script for AI-Powered Document Intelligence & RAG System.
Initializes knowledge base and starts FastAPI / Uvicorn server on http://127.0.0.1:8000
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
    print(" [*] Web UI & SSE Stream: http://127.0.0.1:8000")
    print("========================================================\n")

    uvicorn.run(
        "backend.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )
