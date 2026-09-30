"""
Chunking Strategies for Document Intelligence & RAG.
Provides Semantic & Header-Aware, Sliding Window with Overlap, and Sentence-Boundary chunkers.
"""

import re
import uuid
from typing import List, Dict, Any, Optional


class Chunk:
    """Represents an indexed chunk of text from a parent document."""
    def __init__(
        self,
        chunk_id: str,
        doc_id: str,
        doc_title: str,
        text: str,
        char_start: int,
        char_end: int,
        section_title: str,
        chunk_index: int,
        extra_metadata: Optional[Dict[str, Any]] = None
    ):
        self.chunk_id = chunk_id
        self.doc_id = doc_id
        self.doc_title = doc_title
        self.text = text.strip()
        self.char_start = char_start
        self.char_end = char_end
        self.section_title = section_title
        self.chunk_index = chunk_index
        self.word_count = len(re.findall(r"\b\w+\b", self.text))
        self.extra_metadata = extra_metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "doc_title": self.doc_title,
            "text": self.text,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "section_title": self.section_title,
            "chunk_index": self.chunk_index,
            "word_count": self.word_count,
            "extra_metadata": self.extra_metadata
        }


class DocumentChunker:
    """Orchestrates chunking using configurable strategies."""

    @staticmethod
    def chunk_document(
        doc_id: str,
        doc_title: str,
        text: str,
        strategy: str = "semantic",
        chunk_size: int = 600,
        chunk_overlap: int = 120
    ) -> List[Chunk]:
        """
        Split document text into chunks.
        Strategies: 'semantic', 'sliding_window', 'sentence'
        """
        if strategy == "sliding_window":
            return DocumentChunker._sliding_window_chunk(doc_id, doc_title, text, chunk_size, chunk_overlap)
        elif strategy == "sentence":
            return DocumentChunker._sentence_chunk(doc_id, doc_title, text, chunk_size)
        else: # Default: semantic / structure-aware
            return DocumentChunker._semantic_chunk(doc_id, doc_title, text, chunk_size)

    @staticmethod
    def _semantic_chunk(doc_id: str, doc_title: str, text: str, max_chars: int = 600) -> List[Chunk]:
        """
        Structure-aware chunking:
        Splits text along markdown headers and major section breaks first,
        then groups paragraphs without exceeding max_chars.
        """
        chunks: List[Chunk] = []
        lines = text.split("\n")
        
        current_section = doc_title
        current_buffer: List[str] = []
        current_char_start = 0
        running_char_pos = 0
        chunk_idx = 0

        def flush_buffer():
            nonlocal current_buffer, current_char_start, running_char_pos, chunk_idx
            if not current_buffer:
                return
            combined_text = "\n".join(current_buffer).strip()
            if combined_text:
                c_id = f"{doc_id}_c{chunk_idx}"
                chunks.append(Chunk(
                    chunk_id=c_id,
                    doc_id=doc_id,
                    doc_title=doc_title,
                    text=combined_text,
                    char_start=current_char_start,
                    char_end=current_char_start + len(combined_text),
                    section_title=current_section,
                    chunk_index=chunk_idx
                ))
                chunk_idx += 1
            current_buffer = []

        resume_headers_regex = re.compile(
            r"^(?:technical\s+|core\s+|key\s+|professional\s+)?(?:skills|technical\s+skills|projects|work\s+experience|experience|employment|education|certifications?|licenses?|summary|professional\s+summary|profile|achievements|awards|publications|coursework|academic\s+background|languages|interests)[\s:]*$",
            re.IGNORECASE
        )

        for line in lines:
            line_len = len(line) + 1  # include newline
            line_s = line.strip()
            header_match = re.match(r"^(#{1,4})\s+(.+)$", line_s)
            numbered_match = re.match(r"^(\d+\.\d*(?:\.\d+)*)\s+([A-Z].+)$", line_s)
            resume_header_match = resume_headers_regex.match(line_s)
            
            # Check for uppercase or colon section titles (e.g., "CERTIFICATIONS:", "KEY PROJECTS:")
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

            # Check if this line is a heading
            if header_match or (numbered_match and len(line_s) < 80) or is_custom_header:
                # Flush previous content
                flush_buffer()
                if header_match:
                    current_section = header_match.group(2).strip().title()
                elif numbered_match:
                    current_section = line_s.title()
                elif is_custom_header:
                    current_section = custom_title
                
                current_char_start = running_char_pos
                current_buffer.append(line)
            else:
                # Check if buffer + line exceeds max_chars
                current_len = sum(len(l) + 1 for l in current_buffer)
                if current_len + len(line) > max_chars and current_buffer:
                    flush_buffer()
                    current_char_start = running_char_pos
                
                if not current_buffer:
                    current_char_start = running_char_pos
                current_buffer.append(line)

            running_char_pos += line_len

        flush_buffer()

        # If document produced 0 chunks (empty text), return empty list
        return chunks

    @staticmethod
    def _sliding_window_chunk(
        doc_id: str,
        doc_title: str,
        text: str,
        chunk_size: int = 500,
        chunk_overlap: int = 100
    ) -> List[Chunk]:
        """Fixed character sliding window with configurable overlap."""
        chunks: List[Chunk] = []
        if not text.strip():
            return chunks

        step = max(50, chunk_size - chunk_overlap)
        start = 0
        chunk_idx = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + chunk_size, text_len)
            
            # Snap to word boundary if not at end
            if end < text_len:
                last_space = text.rfind(" ", start, end)
                if last_space != -1 and last_space > start + (chunk_size // 2):
                    end = last_space

            chunk_content = text[start:end].strip()
            if chunk_content:
                c_id = f"{doc_id}_c{chunk_idx}"
                chunks.append(Chunk(
                    chunk_id=c_id,
                    doc_id=doc_id,
                    doc_title=doc_title,
                    text=chunk_content,
                    char_start=start,
                    char_end=end,
                    section_title=f"{doc_title} (Part {chunk_idx + 1})",
                    chunk_index=chunk_idx
                ))
                chunk_idx += 1

            if end >= text_len:
                break
            start += step

        return chunks

    @staticmethod
    def _sentence_chunk(doc_id: str, doc_title: str, text: str, max_chars: int = 600) -> List[Chunk]:
        """Group full sentences together up to max_chars."""
        chunks: List[Chunk] = []
        sentences = re.split(r"(?<=[.!?])\s+", text)
        
        current_sentences: List[str] = []
        current_char_start = 0
        running_char_pos = 0
        chunk_idx = 0

        for s in sentences:
            s_clean = s.strip()
            if not s_clean:
                continue

            current_len = sum(len(st) + 1 for st in current_sentences)
            if current_len + len(s_clean) > max_chars and current_sentences:
                chunk_text = " ".join(current_sentences)
                c_id = f"{doc_id}_c{chunk_idx}"
                chunks.append(Chunk(
                    chunk_id=c_id,
                    doc_id=doc_id,
                    doc_title=doc_title,
                    text=chunk_text,
                    char_start=current_char_start,
                    char_end=current_char_start + len(chunk_text),
                    section_title=f"{doc_title} (Section {chunk_idx + 1})",
                    chunk_index=chunk_idx
                ))
                chunk_idx += 1
                current_sentences = []
                current_char_start = running_char_pos

            current_sentences.append(s_clean)
            running_char_pos += len(s) + 1

        if current_sentences:
            chunk_text = " ".join(current_sentences)
            c_id = f"{doc_id}_c{chunk_idx}"
            chunks.append(Chunk(
                chunk_id=c_id,
                doc_id=doc_id,
                doc_title=doc_title,
                text=chunk_text,
                char_start=current_char_start,
                char_end=current_char_start + len(chunk_text),
                section_title=f"{doc_title} (Section {chunk_idx + 1})",
                chunk_index=chunk_idx
            ))

        return chunks
