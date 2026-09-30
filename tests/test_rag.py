"""
Unit & Integration Tests for Claude-Style Document Intelligence & RAG System.
"""

import unittest
import asyncio
from backend.document_parser import DocumentParser
from backend.chunking import DocumentChunker, Chunk
from backend.retrieval_engine import HybridRetrievalEngine
from backend.rag_pipeline import RAGPipeline
from backend.sample_corpus import SAMPLE_DOCUMENTS


class TestDocumentIntelligenceRAG(unittest.TestCase):

    def test_document_parser_text_and_intelligence(self):
        sample = SAMPLE_DOCUMENTS[0]
        parsed = DocumentParser.parse_text(
            filename=sample["filename"],
            text=sample["content"],
            doc_type="markdown"
        )
        self.assertIn("filename", parsed)
        self.assertIn("metadata", parsed)
        meta = parsed["metadata"]
        
        self.assertGreater(meta["word_count"], 100)
        self.assertGreater(len(meta["summary"]), 20)
        self.assertIn("entities", meta)
        self.assertTrue(len(meta["entities"]["currencies"]) > 0 or len(meta["entities"]["percentages"]) > 0)
        self.assertGreater(len(meta["sections"]), 0)

    def test_document_parser_csv(self):
        csv_bytes = b"Metric,2024,2025\nRevenue,$10M,$15M\nEBITDA,$2M,$4M"
        parsed = DocumentParser.parse_bytes("report.csv", csv_bytes)
        self.assertEqual(parsed["metadata"]["doc_type"], "csv")
        self.assertIn("| Metric | 2024 | 2025 |", parsed["text"])

    def test_chunking_strategies(self):
        text = """# Executive Summary
AlphaTech achieved record profitability in fiscal year 2025 with $1.42B in total revenue.

## Financial Performance
Subscription revenue grew 43.9% year-over-year. GAAP gross margin reached 72.4%.
Operating expenses remained disciplined throughout all four quarters.

## Debt & Liquidity
The company holds $645M in cash and equivalents."""
        
        # 1. Semantic Chunking
        chunks_sem = DocumentChunker.chunk_document("doc_test", "AlphaTech", text, strategy="semantic", chunk_size=300)
        self.assertGreater(len(chunks_sem), 0)
        self.assertIn("Executive Summary", chunks_sem[0].section_title)

        # 2. Sliding Window Chunking
        chunks_slide = DocumentChunker.chunk_document("doc_test", "AlphaTech", text, strategy="sliding_window", chunk_size=150, chunk_overlap=30)
        self.assertGreater(len(chunks_slide), 1)

        # 3. Sentence Chunking
        chunks_sent = DocumentChunker.chunk_document("doc_test", "AlphaTech", text, strategy="sentence", chunk_size=200)
        self.assertGreater(len(chunks_sent), 0)

    def test_hybrid_retrieval_and_rrf(self):
        all_chunks = []
        for idx, sample in enumerate(SAMPLE_DOCUMENTS):
            c_list = DocumentChunker.chunk_document(
                doc_id=f"doc_{idx}",
                doc_title=sample["title"],
                text=sample["content"],
                strategy="semantic",
                chunk_size=500
            )
            all_chunks.extend(c_list)

        engine = HybridRetrievalEngine(rrf_k=60)
        engine.index_chunks(all_chunks)

        # Query 1: Financial revenue
        res_fin = engine.hybrid_search("What was total revenue and GAAP gross margin?", top_k=4)
        self.assertGreater(len(res_fin["results"]), 0)
        top_fin_chunk = res_fin["results"][0]["chunk"]
        self.assertIn("AlphaTech", top_fin_chunk.doc_title)
        self.assertGreater(res_fin["results"][0]["rerank_score"], 0.1)

        # Query 2: SLA uptime and liability cap
        res_sla = engine.hybrid_search("What is the uptime commitment percentage and liability cap?", top_k=4)
        self.assertGreater(len(res_sla["results"]), 0)
        top_sla_chunk = res_sla["results"][0]["chunk"]
        self.assertIn("SaaS", top_sla_chunk.doc_title)

    def test_rag_pipeline_stream_execution(self):
        async def run_stream():
            all_chunks = []
            for idx, sample in enumerate(SAMPLE_DOCUMENTS):
                c_list = DocumentChunker.chunk_document(
                    doc_id=f"doc_{idx}",
                    doc_title=sample["title"],
                    text=sample["content"],
                    strategy="semantic",
                    chunk_size=500
                )
                all_chunks.extend(c_list)

            engine = HybridRetrievalEngine()
            engine.index_chunks(all_chunks)
            pipeline = RAGPipeline(engine)

            tokens = []
            events = []
            async for ev in pipeline.execute_rag_stream("What are the Phase III clinical trial endpoints for BC-809?", top_k=3):
                events.append(ev)
                if ev["type"] == "token":
                    tokens.append(ev["token"])

            full_text = "".join(tokens)
            self.assertGreater(len(full_text), 50)
            self.assertTrue(any(e["type"] == "done" for e in events))

        asyncio.run(run_stream())


    def test_resume_indexing_and_skills_query(self):
        resume_text = """# Bharani - Data Analyst Resume

## SUMMARY
Data Analyst with background in data visualization, SQL, and Full Stack development.

## SKILLS
• Python, SQL, Tableau, Power BI, DAX, Excel
• Full Stack development, API integration, Git, Vercel

## PROJECTS
• IMDb Movie Analytics Dashboard - Power BI | Power Query | DAX | Excel
• Real-world data analysis projects and dashboards

## CERTIFICATIONS
• Google Data Analytics Professional Certificate – Coursera

## EDUCATION
• HSC | Eden Gardens Matric Hr Sec School, Perambalur | Percentage: 83%"""

        parsed = DocumentParser.parse_text("resume.md", resume_text, doc_type="markdown")
        chunks = DocumentChunker.chunk_document("doc_resume", "Bharani Resume", parsed["text"], strategy="semantic", chunk_size=400)
        
        # Check section detection
        section_titles = [c.section_title for c in chunks]
        self.assertTrue(any("Skills" in st for st in section_titles))
        self.assertTrue(any("Projects" in st for st in section_titles))

        engine = HybridRetrievalEngine()
        engine.index_chunks(chunks)

        # Test skills search
        search_res = engine.hybrid_search("what is skills in this document", top_k=3)
        self.assertGreater(len(search_res["results"]), 0)

        pipeline = RAGPipeline(engine)
        
        async def check_query():
            tokens = []
            async for ev in pipeline.execute_rag_stream("what is skills in this document", top_k=3):
                if ev["type"] == "token":
                    tokens.append(ev["token"])
            ans = "".join(tokens)
            self.assertIn("Python", ans)
            self.assertIn("Tableau", ans)
            self.assertIn("Key Skills", ans)

        asyncio.run(check_query())


if __name__ == "__main__":
    unittest.main()
