"""
Document Parser & Document Intelligence Extractor.
Supports PDF, Markdown, Plain Text, CSV, JSON, and extracts rich structural metadata,
key topics, named entities, and structural sections.
"""

import io
import re
import os
import json
import csv
from typing import Dict, List, Any, Optional
from pypdf import PdfReader


class DocumentParser:
    """Parses various document types and extracts intelligence metadata."""

    @staticmethod
    def parse_bytes(filename: str, content: bytes) -> Dict[str, Any]:
        """Parse raw document bytes based on file extension."""
        ext = os.path.splitext(filename)[1].lower()

        if ext == ".pdf":
            return DocumentParser._parse_pdf(filename, content)
        elif ext in [".txt", ".md", ".markdown"]:
            text = content.decode("utf-8", errors="replace")
            return DocumentParser._build_doc_dict(filename, text, doc_type="markdown" if ext in [".md", ".markdown"] else "text")
        elif ext == ".csv":
            text = DocumentParser._parse_csv(content)
            return DocumentParser._build_doc_dict(filename, text, doc_type="csv")
        elif ext == ".json":
            text = DocumentParser._parse_json(content)
            return DocumentParser._build_doc_dict(filename, text, doc_type="json")
        else:
            # Fallback text decoder
            text = content.decode("utf-8", errors="replace")
            return DocumentParser._build_doc_dict(filename, text, doc_type="raw_text")

    @staticmethod
    def parse_text(filename: str, text: str, doc_type: str = "text") -> Dict[str, Any]:
        """Parse raw text string."""
        return DocumentParser._build_doc_dict(filename, text, doc_type=doc_type)

    @staticmethod
    def _parse_pdf(filename: str, content: bytes) -> Dict[str, Any]:
        """Extract text and metadata from PDF bytes."""
        stream = io.BytesIO(content)
        reader = PdfReader(stream)
        num_pages = len(reader.pages)

        pages_text: List[str] = []
        for i, page in enumerate(reader.pages):
            page_content = page.extract_text() or ""
            pages_text.append(page_content.strip())

        full_text = "\n\n".join(pages_text)
        
        pdf_metadata = {}
        if reader.metadata:
            pdf_metadata = {
                "title": reader.metadata.title,
                "author": reader.metadata.author,
                "subject": reader.metadata.subject,
                "creator": reader.metadata.creator,
            }

        doc = DocumentParser._build_doc_dict(
            filename, 
            full_text, 
            doc_type="pdf", 
            extra_meta={
                "page_count": num_pages,
                "pdf_metadata": pdf_metadata,
                "pages": [{"page_number": idx + 1, "text": pt} for idx, pt in enumerate(pages_text)]
            }
        )
        return doc

    @staticmethod
    def _parse_csv(content: bytes) -> str:
        """Convert CSV bytes into structured readable markdown table format."""
        text_data = content.decode("utf-8", errors="replace")
        reader = csv.reader(io.StringIO(text_data))
        rows = list(reader)
        if not rows:
            return ""
        
        header = rows[0]
        lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
        for row in rows[1:]:
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines)

    @staticmethod
    def _parse_json(content: bytes) -> str:
        """Format JSON content as structured readable text."""
        try:
            data = json.loads(content.decode("utf-8", errors="replace"))
            return json.dumps(data, indent=2)
        except Exception:
            return content.decode("utf-8", errors="replace")

    @staticmethod
    def _build_doc_dict(filename: str, text: str, doc_type: str, extra_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate full document intelligence dictionary."""
        words = re.findall(r"\b\w+\b", text)
        word_count = len(words)
        char_count = len(text)
        reading_time_minutes = max(1, round(word_count / 200))

        # Extract entities, keywords, summary, sections
        entities = DocumentParser._extract_entities(text)
        sections = DocumentParser._extract_sections(text)
        summary = DocumentParser._generate_summary(text)
        title = DocumentParser._extract_title(filename, text)

        metadata: Dict[str, Any] = {
            "filename": filename,
            "title": title,
            "doc_type": doc_type,
            "word_count": word_count,
            "char_count": char_count,
            "reading_time_minutes": reading_time_minutes,
            "summary": summary,
            "entities": entities,
            "sections": sections,
            "keywords": DocumentParser._extract_keywords(text),
        }
        if extra_meta:
            metadata.update(extra_meta)

        return {
            "filename": filename,
            "text": text,
            "metadata": metadata
        }

    @staticmethod
    def _extract_title(filename: str, text: str) -> str:
        """Extract title from first H1 or document header, fallback to filename."""
        first_lines = text.strip().split("\n")[:5]
        for line in first_lines:
            line_clean = line.strip()
            if line_clean.startswith("# ") and len(line_clean) > 3:
                return line_clean.replace("# ", "").strip()
            if line_clean.isupper() and 5 < len(line_clean) < 80:
                return line_clean.title()
        
        # Fallback to sanitized filename
        base = os.path.splitext(filename)[0]
        return base.replace("_", " ").replace("-", " ").title()

    @staticmethod
    def _generate_summary(text: str, max_sentences: int = 3) -> str:
        """Extract key informative sentences for document preview."""
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        meaningful = [s.strip() for s in sentences if len(s.strip()) > 30 and not s.strip().startswith("#")]
        if not meaningful:
            return text[:200] + "..." if len(text) > 200 else text
        return " ".join(meaningful[:max_sentences])

    @staticmethod
    def _extract_entities(text: str) -> Dict[str, List[str]]:
        """Extract named entities including currency, percentages, dates, metrics, and key clauses."""
        # Currencies (e.g., $1.2B, $450,000, €50M, USD 10M)
        currencies = list(set(re.findall(r"(?:\$|€|£|USD|EUR)\s?\d+(?:,\d{3})*(?:\.\d+)?(?:\s?(?:billion|million|trillion|[BMKbmk]))?", text)))
        
        # Percentages (e.g., 99.9%, 14.5%, 35 percent)
        percentages = list(set(re.findall(r"\b\d+(?:\.\d+)?\s?(?:%|percent)\b", text, re.IGNORECASE)))
        
        # Dates and Quarters (e.g., Q3 2025, 2024-2025, January 15, 2025)
        dates = list(set(re.findall(r"\b(?:Q[1-4]\s?\d{4}|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}|\d{4}-\d{2}-\d{2}|\b202\d\b)", text)))
        
        # Key Technical & Legal Terms
        technical_matches = list(set(re.findall(
            r"\b(?:SLA|EBITDA|FDA|EMA|CRF|HIPAA|SOC-2|GDPR|API|RAG|LLM|PFS|OS|ORR|RECIST|Indemnification|Force Majeure|Covenant|Sublinear|Vector Space|PCA|t-SNE|BM25|RRF)\b", 
            text, 
            re.IGNORECASE
        )))

        # Organizations / Systems (Capitalized Multi-word phrases)
        orgs = list(set(re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b", text)))
        filtered_orgs = [o for o in orgs if len(o.split()) in [2, 3] and not o.startswith(("This", "That", "There", "When", "What", "Table", "Figure"))][:8]

        return {
            "currencies": sorted(currencies[:8]),
            "percentages": sorted(percentages[:8]),
            "dates": sorted(dates[:8]),
            "technical_terms": sorted(list(set([t.upper() for t in technical_matches])))[:12],
            "organizations": sorted(filtered_orgs)
        }

    @staticmethod
    def _extract_sections(text: str) -> List[Dict[str, Any]]:
        """Detect markdown headings, numbered sections, and resume/document section headers."""
        sections = []
        lines = text.split("\n")
        char_start = 0

        resume_headers_regex = re.compile(
            r"^(?:technical\s+|core\s+|key\s+|professional\s+)?(?:skills|technical\s+skills|projects|work\s+experience|experience|employment|education|certifications?|licenses?|summary|professional\s+summary|profile|achievements|awards|publications|coursework|academic\s+background|languages|interests)[\s:]*$",
            re.IGNORECASE
        )

        for line in lines:
            line_s = line.strip()
            header_match = re.match(r"^(#{1,4})\s+(.+)$", line_s)
            numbered_match = re.match(r"^(\d+\.\d*(?:\.\d+)*)\s+([A-Z].+)$", line_s)
            resume_header_match = resume_headers_regex.match(line_s)

            is_custom_header = False
            custom_title = ""
            if resume_header_match:
                is_custom_header = True
                custom_title = line_s.rstrip(":").strip().title()
            elif 3 <= len(line_s) <= 40 and not line_s.startswith(("-", "*", "•", "–")) and (
                (line_s.isupper() and any(c.isalpha() for c in line_s)) or
                (line_s.endswith(":") and not any(ch in line_s for ch in ["http", "{", "}", "[", "]"]))
            ):
                is_custom_header = True
                custom_title = line_s.rstrip(":").strip().title()

            if header_match:
                level = len(header_match.group(1))
                title = header_match.group(2).strip()
                sections.append({"title": title, "level": level, "start_char": char_start})
            elif numbered_match and len(line_s) < 80:
                sections.append({"title": line_s, "level": 2, "start_char": char_start})
            elif is_custom_header:
                sections.append({"title": custom_title, "level": 2, "start_char": char_start})
                
            char_start += len(line) + 1

        if not sections:
            sections.append({"title": "General Document", "level": 1, "start_char": 0})

        return sections[:25]

    @staticmethod
    def _extract_keywords(text: str, top_n: int = 10) -> List[str]:
        """Extract salient keywords based on word frequency excluding stop words."""
        stop_words = {
            "the", "and", "a", "to", "of", "in", "is", "that", "for", "it", "as", "was", "with", 
            "on", "by", "at", "an", "be", "this", "which", "or", "from", "are", "not", "have", 
            "has", "we", "our", "you", "your", "their", "they", "will", "shall", "all", "any", 
            "can", "such", "than", "other", "into", "more", "also", "including", "per", "due"
        }
        words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
        freq: Dict[str, int] = {}
        for w in words:
            if w not in stop_words:
                freq[w] = freq.get(w, 0) + 1
        
        sorted_kw = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [w for w, _ in sorted_kw[:top_n]]
