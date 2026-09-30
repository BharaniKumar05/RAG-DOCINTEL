"""
Claude-Style Grounded RAG Pipeline Orchestrator.
Synthesizes direct, unified, executive, and natural answers strictly from verified context documents.
No introductory boilerplate or repetitive document mentions.
"""

import os
import time
import json
import re
import asyncio
from typing import List, Dict, Any, AsyncGenerator, Optional
import google.generativeai as genai
import openai

from backend.chunking import Chunk
from backend.retrieval_engine import HybridRetrievalEngine


class RAGPipeline:
    """Orchestrates Hybrid RAG queries, prompt construction, and Claude-style streaming generation."""

    def __init__(self, retrieval_engine: HybridRetrievalEngine):
        self.retrieval_engine = retrieval_engine
        self.active_provider = os.getenv("RAG_LLM_PROVIDER", "local")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.model_name = os.getenv("RAG_MODEL_NAME", "Claude-Style Local Engine")

    def configure_provider(self, provider: str, api_key: str = "", model_name: str = ""):
        """Update active LLM provider and credentials."""
        self.active_provider = provider
        if provider == "gemini":
            self.gemini_api_key = api_key or self.gemini_api_key
            self.model_name = model_name or "gemini-2.5-flash"
            if self.gemini_api_key:
                genai.configure(api_key=self.gemini_api_key)
        elif provider == "openai":
            self.openai_api_key = api_key or self.openai_api_key
            self.model_name = model_name or "gpt-4o"
        else:
            self.active_provider = "local"
            self.model_name = "Claude-Style Local Engine"

    async def execute_rag_stream(
        self,
        query: str,
        top_k: int = 4,
        temperature: float = 0.2,
        doc_filter: Optional[str] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Execute RAG search and stream a clean, unified, Claude-style answer without boilerplate intros.
        """
        start_time = time.time()

        # Hybrid Search with optional document filter
        search_result = self.retrieval_engine.hybrid_search(query, top_k=top_k, rerank=True, doc_filter=doc_filter)
        ranked_results = search_result.get("results", [])

        citations: List[Dict[str, Any]] = []
        context_blocks: List[str] = []

        for idx, item in enumerate(ranked_results, start=1):
            chunk: Chunk = item["chunk"]
            citation_info = {
                "source_index": idx,
                "doc_id": chunk.doc_id,
                "doc_title": chunk.doc_title,
                "section_title": chunk.section_title,
                "text": chunk.text,
                "confidence_percent": item.get("confidence_percent", 85)
            }
            citations.append(citation_info)
            context_blocks.append(f"[{chunk.doc_title} | {chunk.section_title}]:\n{chunk.text}")

        if not citations:
            yield {
                "type": "token",
                "token": "I couldn't find any relevant information in the uploaded documents to answer your question. Please ensure the relevant document is uploaded or try rephrasing your query."
            }
            yield {
                "type": "done",
                "telemetry": {
                    "total_latency_ms": round((time.time() - start_time) * 1000, 2),
                    "citations": []
                }
            }
            return

        grounded_context = "\n\n---\n\n".join(context_blocks)
        
        # Stream from active provider
        if self.active_provider == "gemini" and self.gemini_api_key:
            async for token in self._stream_gemini(query, grounded_context):
                yield {"type": "token", "token": token}
        elif self.active_provider == "openai" and self.openai_api_key:
            async for token in self._stream_openai(query, grounded_context):
                yield {"type": "token", "token": token}
        else:
            async for token in self._stream_claude_style_local(query, citations):
                yield {"type": "token", "token": token}

        total_latency = round((time.time() - start_time) * 1000, 2)
        yield {
            "type": "done",
            "telemetry": {
                "total_latency_ms": total_latency,
                "citations": citations,
                "model": self.model_name
            }
        }

    async def _stream_gemini(self, query: str, context: str) -> AsyncGenerator[str, None]:
        prompt = f"""You are Claude, a helpful, precise, and thoughtful AI assistant.
Answer the user's question directly, clearly, and comprehensively using ONLY the provided verified document excerpts.
Rules:
- Give a single, cohesive, natural answer.
- DO NOT begin with boilerplate like "Based on [document name]..." or "According to the provided document...". Start immediately with the direct factual answer.
- Format with clean markdown paragraphs, bullet points, or concise tables if appropriate.
- Only use facts present in the excerpts. Do not speculate.

DOCUMENT EXCERPTS:
{context}

QUESTION:
{query}

ANSWER:"""
        try:
            model = genai.GenerativeModel(self.model_name or "gemini-2.5-flash")
            response = await asyncio.to_thread(model.generate_content, prompt, stream=True)
            for chunk in response:
                if chunk.text:
                    yield chunk.text
                    await asyncio.sleep(0.01)
        except Exception:
            pass

    async def _stream_openai(self, query: str, context: str) -> AsyncGenerator[str, None]:
        messages = [
            {
                "role": "system",
                "content": "You are Claude, a helpful, precise, and thoughtful AI assistant. Answer directly and comprehensively using only the provided document excerpts. Do not start with boilerplate like 'Based on the document...'."
            },
            {
                "role": "user",
                "content": f"EXCERPTS:\n{context}\n\nQUESTION: {query}"
            }
        ]
        try:
            client = openai.OpenAI(api_key=self.openai_api_key)
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model=self.model_name or "gpt-4o",
                messages=messages,
                stream=True,
                temperature=0.2
            )
            for chunk in response:
                delta = chunk.choices[0].delta.content if chunk.choices else ""
                if delta:
                    yield delta
                    await asyncio.sleep(0.01)
        except Exception:
            pass

    async def _stream_claude_style_local(
        self,
        query: str,
        citations: List[Dict[str, Any]]
    ) -> AsyncGenerator[str, None]:
        """
        Synthesizes a direct, query-targeted, high quality structured answer.
        """
        await asyncio.sleep(0.02)

        q_lower = query.lower()
        stop_words = {
            "what", "is", "are", "the", "and", "a", "an", "in", "to", "for", "of", "with", "on", 
            "under", "about", "how", "why", "from", "by", "that", "this", "or", "tell", "me", "show", "give",
            "document", "documents", "file", "pdf", "text", "page", "please", "can", "you", "details"
        }
        q_words = [w.lower() for w in re.findall(r"\b\w+\b", query) if w.lower() not in stop_words]

        # Classify user query intent
        is_skills_query = any(k in q_lower for k in ["skill", "stack", "language", "tech", "tool", "expertise", "competenc"])
        is_projects_query = any(k in q_lower for k in ["project", "portfolio", "built", "dashboard", "developed", "app"])
        is_certs_query = any(k in q_lower for k in ["certif", "coursera", "license", "course", "credential"])
        is_edu_query = any(k in q_lower for k in ["educat", "school", "college", "degree", "hsc", "sslc", "percent", "grade", "gpa", "btech", "bsc", "university"])
        is_exp_query = any(k in q_lower for k in ["experience", "work", "role", "job", "intern", "company", "career"])

        # Extract structured content from matching chunks
        sections_map: Dict[str, List[str]] = {}
        table_lines: List[str] = []
        general_lines: List[str] = []
        seen_lines = set()

        for c in citations:
            sec_title = c.get("section_title", "General Information")
            chunk_text = c.get("text", "")
            lines = chunk_text.split("\n")
            
            for line in lines:
                l_str = line.strip()
                if not l_str:
                    continue
                if "|" in l_str and not l_str.startswith("http"):
                    if l_str not in table_lines:
                        table_lines.append(l_str)
                    continue

                clean_norm = re.sub(r"[^\w\s]", "", l_str.lower())
                if clean_norm in seen_lines:
                    continue
                seen_lines.add(clean_norm)

                # Skip header lines inside section grouping
                if l_str.startswith("#"):
                    continue

                if sec_title not in sections_map:
                    sections_map[sec_title] = []
                sections_map[sec_title].append(l_str)
                general_lines.append((l_str, sec_title))

        blocks: List[str] = []

        # 1. Targeted Intent Synthesis
        if is_skills_query:
            skill_lines = []
            for sec, lines in sections_map.items():
                sec_l = sec.lower()
                if any(k in sec_l for k in ["skill", "tech", "competenc", "language", "tool", "expertise"]):
                    skill_lines.extend(lines)
            
            # If no dedicated skills section found, search all lines for skill terms
            if not skill_lines:
                for l_str, sec in general_lines:
                    l_lower = l_str.lower()
                    if any(w in l_lower for w in ["python", "sql", "tableau", "power bi", "dax", "excel", "javascript", "react", "git", "api", "html", "css", "data", "machine learning", "analysis"]):
                        skill_lines.append(l_str)

            if skill_lines:
                formatted_skills = []
                for s in skill_lines:
                    clean_s = s.lstrip("-* •–\t").strip()
                    if clean_s:
                        if ":" in clean_s and len(clean_s.split(":")[0]) < 35:
                            parts = clean_s.split(":", 1)
                            formatted_skills.append(f"- **{parts[0].strip()}**: {parts[1].strip()}")
                        else:
                            formatted_skills.append(f"- {clean_s}")
                blocks.append("### Key Skills & Technical Proficiencies\n" + "\n".join(formatted_skills))

        elif is_projects_query:
            project_lines = []
            for sec, lines in sections_map.items():
                sec_l = sec.lower()
                if any(k in sec_l for k in ["project", "portfolio", "development", "dashboard"]):
                    project_lines.extend(lines)
            
            if not project_lines:
                for l_str, sec in general_lines:
                    l_lower = l_str.lower()
                    if any(w in l_lower for w in ["project", "dashboard", "analytics", "developed", "built", "implemented"]):
                        project_lines.append(l_str)

            if project_lines:
                formatted_projects = []
                for p in project_lines:
                    clean_p = p.lstrip("-* •–\t").strip()
                    if clean_p:
                        if "|" in clean_p:
                            formatted_projects.append(f"- **{clean_p}**")
                        elif ":" in clean_p and len(clean_p.split(":")[0]) < 35:
                            parts = clean_p.split(":", 1)
                            formatted_projects.append(f"- **{parts[0].strip()}**: {parts[1].strip()}")
                        else:
                            formatted_projects.append(f"- {clean_p}")
                blocks.append("### Projects & Portfolios\n" + "\n".join(formatted_projects))

        elif is_certs_query:
            cert_lines = []
            for sec, lines in sections_map.items():
                sec_l = sec.lower()
                if any(k in sec_l for k in ["certif", "course", "coursera", "license"]):
                    cert_lines.extend(lines)
            if not cert_lines:
                for l_str, sec in general_lines:
                    if any(w in l_str.lower() for w in ["certificat", "coursera", "google", "credential", "license"]):
                        cert_lines.append(l_str)
            if cert_lines:
                formatted_certs = []
                for c in cert_lines:
                    cl = c.lstrip("-* •–\t").strip()
                    if cl:
                        formatted_certs.append(f"- {cl}")
                blocks.append("### Certifications & Courses\n" + "\n".join(formatted_certs))

        elif is_edu_query:
            edu_lines = []
            for sec, lines in sections_map.items():
                sec_l = sec.lower()
                if any(k in sec_l for k in ["educat", "academic", "school", "college"]):
                    edu_lines.extend(lines)
            if not edu_lines:
                for l_str, sec in general_lines:
                    if any(w in l_str.lower() for w in ["school", "college", "university", "percentage", "hsc", "sslc", "degree", "btech"]):
                        edu_lines.append(l_str)
            if edu_lines:
                formatted_edu = []
                for e in edu_lines:
                    el = e.lstrip("-* •–\t").strip()
                    if el:
                        formatted_edu.append(f"- {el}")
                blocks.append("### Educational Background\n" + "\n".join(formatted_edu))

        elif is_exp_query:
            exp_lines = []
            for sec, lines in sections_map.items():
                sec_l = sec.lower()
                if any(k in sec_l for k in ["experience", "work", "employment", "career", "role"]):
                    exp_lines.extend(lines)
            if exp_lines:
                formatted_exp = []
                for e in exp_lines:
                    el = e.lstrip("-* •–\t").strip()
                    if el:
                        formatted_exp.append(f"- {el}")
                blocks.append("### Work Experience & Roles\n" + "\n".join(formatted_exp))

        # 2. If no intent matched or blocks empty, perform general smart synthesis
        if not blocks:
            # Score each candidate line based on query keyword matches
            scored_lines = []
            for l_str, sec in general_lines:
                clean_l = l_str.lstrip("-* •–\t").strip()
                if not clean_l or len(clean_l) < 4:
                    continue
                match_count = sum(1 for qw in q_words if qw in clean_l.lower())
                scored_lines.append((clean_l, sec, match_count))

            scored_lines.sort(key=lambda x: x[2], reverse=True)
            
            lead_items = [item for item in scored_lines if item[2] > 0][:5]
            other_items = [item for item in scored_lines if item not in lead_items][:4]
            selected = lead_items + (other_items if not lead_items else [])

            if selected:
                main_answer = []
                for clean_l, sec, _ in selected:
                    if ":" in clean_l and len(clean_l.split(":")[0]) < 35:
                        parts = clean_l.split(":", 1)
                        main_answer.append(f"- **{parts[0].strip()}**: {parts[1].strip()}")
                    else:
                        main_answer.append(f"- {clean_l}")
                blocks.append("\n".join(main_answer))

        # 3. Include markdown tables if present in citations
        if table_lines and len(table_lines) >= 2:
            blocks.append("\n" + "\n".join(table_lines[:10]))

        full_text = "\n\n".join(blocks) if blocks else "I couldn't find specific details for that query in the verified document excerpts."

        # Stream words smoothly
        words = full_text.split(" ")
        for i, word in enumerate(words):
            token = word + (" " if i < len(words) - 1 else "")
            yield token
            await asyncio.sleep(0.012)
