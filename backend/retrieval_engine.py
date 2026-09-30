"""
Hybrid Retrieval Engine:
- Dense Vector Store (TF-IDF sublinear n-gram embeddings / Cosine Similarity)
- Sparse BM25 Inverted Index (Okapi BM25)
- Reciprocal Rank Fusion (RRF)
- Contextual Cross-Encoder Reranker
- 2D Vector Space Projection (PCA / TruncatedSVD)
"""

import re
import math
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

from backend.chunking import Chunk


class HybridRetrievalEngine:
    """Manages dual indices (Dense Vector + BM25) and coordinates hybrid search."""

    def __init__(self, rrf_k: int = 60, dense_weight: float = 0.5, bm25_weight: float = 0.5):
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight

        self.chunks: List[Chunk] = []
        self.chunk_id_map: Dict[str, Chunk] = {}

        # Dense Vectorizer
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.chunk_embeddings: Optional[np.ndarray] = None
        
        # BM25 Index
        self.bm25: Optional[BM25Okapi] = None
        self.bm25_corpus: List[List[str]] = []

        # 2D Projection Model
        self.svd_model: Optional[TruncatedSVD] = None
        self.coords_2d: Dict[str, Tuple[float, float]] = {}

    def index_chunks(self, chunks: List[Chunk]):
        """Index a list of chunks into dense and sparse stores."""
        self.chunks = chunks
        self.chunk_id_map = {c.chunk_id: c for c in chunks}
        self.coords_2d = {}

        if not chunks:
            self.vectorizer = None
            self.chunk_embeddings = None
            self.bm25 = None
            self.bm25_corpus = []
            return

        corpus_texts = [f"{c.doc_title} {c.section_title} {c.text}" for c in chunks]

        # 1. Build Dense TF-IDF Matrix (Sublinear scaling, word 1-2 ngrams)
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            stop_words="english",
            max_features=10000
        )
        self.chunk_embeddings = self.vectorizer.fit_transform(corpus_texts)

        # 2. Build BM25 Index
        tokenized_corpus = [self._tokenize(text) for text in corpus_texts]
        self.bm25_corpus = tokenized_corpus
        self.bm25 = BM25Okapi(tokenized_corpus)

        # 3. Compute 2D SVD/PCA Projection for Vector Space Canvas
        self._compute_2d_projections()

    def _tokenize(self, text: str) -> List[str]:
        """Normalize and tokenize text for BM25."""
        clean_text = re.sub(r"[^\w\s]", " ", text.lower())
        return [w for w in clean_text.split() if len(w) > 1]

    def _tokenize_query(self, query: str) -> List[str]:
        """Extract meaningful query tokens excluding conversational and document boilerplate."""
        stop_words = {
            "what", "is", "are", "the", "and", "a", "an", "in", "to", "for", "of", "with", "on",
            "at", "by", "from", "about", "how", "why", "who", "whom", "this", "that", "these", "those",
            "document", "documents", "doc", "docs", "file", "files", "pdf", "text", "page",
            "tell", "me", "show", "give", "list", "find", "describe", "explain", "summarize",
            "details", "detail", "info", "information", "please", "can", "you", "all", "any"
        }
        tokens = self._tokenize(query)
        filtered = [t for t in tokens if t not in stop_words]
        return filtered if filtered else tokens

    def _compute_2d_projections(self):
        """Project high-dimensional embeddings to 2D coordinates scaled [-100, 100]."""
        if self.chunk_embeddings is None or len(self.chunks) == 0:
            return

        n_samples = len(self.chunks)
        if n_samples < 2:
            self.coords_2d = {self.chunks[0].chunk_id: (0.0, 0.0)}
            return

        n_components = 2
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        try:
            coords = svd.fit_transform(self.chunk_embeddings)
            self.svd_model = svd

            # Normalize coordinates to range [-80, 80]
            max_x = max(abs(coords[:, 0].min()), abs(coords[:, 0].max())) or 1.0
            max_y = max(abs(coords[:, 1].min()), abs(coords[:, 1].max())) or 1.0

            for idx, chunk in enumerate(self.chunks):
                norm_x = float((coords[idx, 0] / max_x) * 80.0)
                norm_y = float((coords[idx, 1] / max_y) * 80.0)
                self.coords_2d[chunk.chunk_id] = (round(norm_x, 2), round(norm_y, 2))
        except Exception:
            # Fallback spread
            for idx, chunk in enumerate(self.chunks):
                angle = (idx / max(1, n_samples)) * 2 * math.pi
                r = 40.0 + (idx % 3) * 15.0
                self.coords_2d[chunk.chunk_id] = (
                    round(r * math.cos(angle), 2),
                    round(r * math.sin(angle), 2)
                )

    def project_query_2d(self, query: str) -> Tuple[float, float]:
        """Project a query into the current 2D coordinate space."""
        if not self.vectorizer or not self.svd_model:
            return (0.0, 0.0)
        try:
            q_vec = self.vectorizer.transform([query])
            coords = self.svd_model.transform(q_vec)[0]
            max_val = max(abs(coords[0]), abs(coords[1])) or 1.0
            x = float((coords[0] / max_val) * 70.0)
            y = float((coords[1] / max_val) * 70.0)
            return (round(x, 2), round(y, 2))
        except Exception:
            return (0.0, 0.0)

    def retrieve_dense(self, query: str, top_k: int = 15, doc_filter: Optional[str] = None) -> List[Tuple[Chunk, float]]:
        """Dense cosine vector search."""
        if not self.vectorizer or self.chunk_embeddings is None or not self.chunks:
            return []

        q_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(q_vec, self.chunk_embeddings).flatten()
        
        ranked_indices = np.argsort(similarities)[::-1]
        results = []
        for idx in ranked_indices:
            chunk = self.chunks[idx]
            if doc_filter and chunk.doc_id != doc_filter:
                continue
            score = float(similarities[idx])
            if score > 0.0005:
                results.append((chunk, score))
            if len(results) >= top_k:
                break
        return results

    def retrieve_bm25(self, query: str, top_k: int = 15, doc_filter: Optional[str] = None) -> List[Tuple[Chunk, float]]:
        """Sparse BM25 lexical keyword search."""
        if not self.bm25 or not self.chunks:
            return []

        tokens = self._tokenize_query(query)
        if not tokens:
            tokens = self._tokenize(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        ranked_indices = np.argsort(scores)[::-1]
        
        results = []
        max_score = max(scores) if len(scores) > 0 and max(scores) > 0 else 1.0
        for idx in ranked_indices:
            chunk = self.chunks[idx]
            if doc_filter and chunk.doc_id != doc_filter:
                continue
            raw_score = float(scores[idx])
            if raw_score > 0.0005:
                norm_score = float(raw_score / max_score)
                results.append((chunk, norm_score))
            if len(results) >= top_k:
                break
        return results

    def hybrid_search(
        self,
        query: str,
        top_k: int = 8,
        rerank: bool = True,
        doc_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute Hybrid Search (Dense + Sparse BM25) fused with Reciprocal Rank Fusion (RRF),
        followed by Contextual Cross-Encoder style Reranking.
        """
        if not self.chunks:
            return {
                "results": [],
                "dense_candidates": [],
                "bm25_candidates": [],
                "query_2d": (0.0, 0.0),
                "total_chunks_searched": 0
            }

        dense_matches = self.retrieve_dense(query, top_k=25, doc_filter=doc_filter)
        bm25_matches = self.retrieve_bm25(query, top_k=25, doc_filter=doc_filter)

        # Reciprocal Rank Fusion
        rrf_scores: Dict[str, float] = {}
        dense_rank_map: Dict[str, int] = {}
        bm25_rank_map: Dict[str, int] = {}
        dense_score_map: Dict[str, float] = {}
        bm25_score_map: Dict[str, float] = {}

        for rank, (chunk, score) in enumerate(dense_matches, start=1):
            dense_rank_map[chunk.chunk_id] = rank
            dense_score_map[chunk.chunk_id] = score
            rrf_val = self.dense_weight * (1.0 / (self.rrf_k + rank))
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + rrf_val

        for rank, (chunk, score) in enumerate(bm25_matches, start=1):
            bm25_rank_map[chunk.chunk_id] = rank
            bm25_score_map[chunk.chunk_id] = score
            rrf_val = self.bm25_weight * (1.0 / (self.rrf_k + rank))
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + rrf_val

        # If one search method found candidates that the other didn't, ensure all candidates are present
        fused_candidates = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:20]

        candidate_records = []
        for c_id, rrf_s in fused_candidates:
            if c_id not in self.chunk_id_map:
                continue
            chunk = self.chunk_id_map[c_id]
            coords = self.coords_2d.get(c_id, (0.0, 0.0))
            candidate_records.append({
                "chunk": chunk,
                "rrf_score": round(rrf_s, 5),
                "dense_rank": dense_rank_map.get(c_id, None),
                "dense_score": round(dense_score_map.get(c_id, 0.0), 4),
                "bm25_rank": bm25_rank_map.get(c_id, None),
                "bm25_score": round(bm25_score_map.get(c_id, 0.0), 4),
                "coords_2d": coords
            })

        # Apply Contextual Reranking
        if rerank and candidate_records:
            final_ranked = self._rerank_candidates(query, candidate_records, top_k=top_k)
        else:
            final_ranked = candidate_records[:top_k]

        query_coords = self.project_query_2d(query)

        return {
            "results": final_ranked,
            "dense_candidates": [
                {"chunk_id": c.chunk_id, "doc_title": c.doc_title, "score": round(s, 4)} 
                for c, s in dense_matches[:5]
            ],
            "bm25_candidates": [
                {"chunk_id": c.chunk_id, "doc_title": c.doc_title, "score": round(s, 4)} 
                for c, s in bm25_matches[:5]
            ],
            "query_2d": query_coords,
            "total_chunks_searched": len(self.chunks)
        }

    def _rerank_candidates(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 6) -> List[Dict[str, Any]]:
        """
        Contextual cross-encoder style reranking:
        Evaluates exact phrase matching, lexical overlap, section heading alignment,
        synonym intent, and dense/BM25 scores to compute calibrated relevance.
        """
        q_clean = query.lower()
        q_tokens = set(self._tokenize_query(query))
        
        # Intent-based section mappings
        intent_synonyms = {
            "skill": ["skill", "skills", "technical skills", "tech stack", "languages", "tools", "technologies", "expertise", "frameworks", "proficiencies", "competencies"],
            "project": ["project", "projects", "portfolio", "dashboard", "dashboards", "development", "built", "implemented", "application"],
            "experience": ["experience", "work experience", "employment", "internship", "career", "history", "role", "responsibilities"],
            "education": ["education", "academic", "degree", "school", "college", "university", "percentage", "gpa", "coursework", "matric"],
            "certification": ["certification", "certifications", "certified", "certificate", "coursera", "license", "credentials"],
            "summary": ["summary", "overview", "profile", "about", "executive summary", "objective"]
        }

        scored_items = []
        for item in candidates:
            chunk: Chunk = item["chunk"]
            text_lower = chunk.text.lower()
            section_lower = chunk.section_title.lower()
            doc_lower = chunk.doc_title.lower()
            
            # 1. Exact phrase match bonus
            phrase_bonus = 0.0
            if len(q_clean) > 3 and q_clean in text_lower:
                phrase_bonus = 0.40

            # 2. Token overlap ratio against text
            chunk_tokens = set(self._tokenize(chunk.text))
            token_overlap = len(q_tokens.intersection(chunk_tokens)) / max(1, len(q_tokens))

            # 3. Section Title / Header relevance bonus
            section_tokens = set(self._tokenize(section_lower))
            doc_tokens = set(self._tokenize(doc_lower))
            header_tokens = section_tokens.union(doc_tokens)
            header_overlap = len(q_tokens.intersection(header_tokens)) / max(1, len(q_tokens))

            # 4. Intent synonym alignment bonus
            intent_bonus = 0.0
            for intent_key, syn_list in intent_synonyms.items():
                if any(syn in q_clean for syn in [intent_key, intent_key + "s"]):
                    # If section matches this intent
                    if any(syn in section_lower for syn in syn_list):
                        intent_bonus += 0.45
                    elif any(syn in text_lower for syn in syn_list):
                        intent_bonus += 0.20

            # 5. Dense & BM25 normalized signals
            dense_s = item.get("dense_score", 0.0)
            bm25_s = item.get("bm25_score", 0.0)

            # Composite Rerank Score
            rerank_score = (
                (token_overlap * 0.25) +
                (header_overlap * 0.25) +
                (intent_bonus * 0.25) +
                (phrase_bonus * 0.15) +
                (dense_s * 0.05) +
                (bm25_s * 0.05)
            )
            # Bound score [0.05, 1.0]
            calibrated_score = min(1.0, max(0.05, rerank_score))

            item_copy = dict(item)
            item_copy["rerank_score"] = round(calibrated_score, 4)
            item_copy["confidence_percent"] = min(99, max(50, int(calibrated_score * 100)))
            scored_items.append(item_copy)

        # Sort by rerank score
        scored_items.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored_items[:top_k]
