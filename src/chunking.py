"""
Document chunking strategies for incident review documents.

This module provides intelligent chunking that respects document structure
while ensuring chunks are appropriately sized for embedding and retrieval.
"""

import re
from dataclasses import dataclass
from typing import Iterator, Optional
import frontmatter
import tiktoken

@dataclass
class Chunk:
    content: str
    metadata: dict
    chunk_index: int
    token_count: int
    section: Optional[str] = None

class IncidentChunker:
    def __init__(
            self,
            max_tokens: int = 512,
            overlap_tokens: int = 50,
            model: str = "gpt-3.5-turbo"  # For tiktoken encoding
        ):

        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.encoder = tiktoken.encoding_for_model(model)

    def count_tokens(self, text: str) -> int:
        return len(self.encoder.encode(text))
    
    def parse_document(self, content: str, filename: str) -> tuple[dict, str]:
        post = frontmatter.loads(content)
        metadata = dict(post.metadata)
        metadata['source_file'] = filename

        return metadata, post.content
    
    def extract_sections(self, content: str) -> list[tuple[Optional[str], str]]:
        pattern = r'^## (.+)$'

        sections = []
        current_section = None
        current_content = []

        for line in content.split('\n'):
            match = re.match(pattern, line)
            if match:
                if current_content:
                    sections.append((current_section, '\n'.join(current_content)))

                    current_section = match.group(1).strip()
                    current_content = []
                else:
                    current_content.append(line)

        if current_content:
            sections.append((current_section, '\n'.join(current_content)))

        return sections
    
    def split_long_section(self, text: str) -> Iterator[str]:
        paragraphs = text.split('\n\n')
        current_chunk = []
        current_tokens = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            para_tokens = self.count_tokens(para)

            if para_tokens > self.max_tokens:
                if current_chunk:
                    yield '\n\n'.join(current_chunk)

                    current_chunk = []
                    current_tokens = 0

                yield from self.split_long_section(para)

            elif current_tokens + para_tokens > self.max_tokens:
                yield '\n\n'.join(current_chunk)
                current_chunk = [para]
                current_tokens = para_tokens
            else:
                current_chunk.append(para)
                current_tokens += para_tokens

        if current_chunk:
            yield '\n\n'.join(current_chunk)

    
    def split_by_sentences(self, text: str) -> Iterator[str]:
        sentences = re.split('(?<=[.!?])\s+', text)

        current_chunk = []
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)

            if current_tokens + sentence_tokens > self.max_tokens:
                if current_chunk:
                    yield ' '.join(current_chunk)
                current_chunk = [sentence]
                current_tokens = sentence_tokens
            else:
                current_tokens.append(sentence)
                current_tokens += sentence_tokens

            if current_chunk:
                yield ' '.join(current_chunk)


    def chunk_document(
            self,
            content: str,
            filename: str
    ) -> list[Chunk]:
        metadata, body = self.parse_document(content, filename)
        sections = self.extract_sections(body)

        chunks = []
        chunk_index = 0

        summary = self.create_summary_chunk(metadata)

        if summary:
            chunks.append(Chunk(
                content=summary,
                metadata=metadata,
                chunk_index=chunk_index,
                token_count=self.count_tokens(summary),
                section="Summary"
            ))
            chunk_index += 1

            for section_name, section_content in sections:
                sections_content = section_content.strip()

                if not section_content:
                    continue

                section_tokens = self.count_tokens(section_content)

                if section_tokens <= self.max_tokens:
                    chunks.append(Chunk(
                        content=section_content,
                        metadata=metadata,
                        chunk_index=chunk_index,
                        token_count=section_tokens,
                        section=section_name
                    ))
                    chunk_index += 1
                else:
                    # need to split section
                    for chunk_content in self.split_long_section(section_content):
                        chunks.append(Chunk(
                            content=chunk_content,
                            metadata=metadata,
                            chunk_index=chunk_index,
                            token_count=self.count_tokens(chunk_content),
                            section=section_name
                        ))
                        chunk_index += 1

            return chunks
        
    def create_summary_chunk(self, metadata: dict) -> Optional[str]:
        parts = []

        if 'title' in metadata:
            parts.append(f"Incidents: {metadata['title']}")

        if 'incident_id' in metadata:
            parts.append(f"ID: {metadata['incident_id']}")

        if 'severity' in metadata:
            parts.append(f"Severity: {metadata['severity']}")

        if 'date' in metadata:
            parts.append(f"Date: {metadata['date']}")

        if 'duration_minutes' in metadata:
            parts.append(f"Duration: {metadata['duration_minutes']} minutes")

        if 'services_affected' in metadata:
            services = metadata['services_affected']

            if isinstance(services, list):
                services = ', '.join(services)

            parts.append(f'Services affected: {services}')

        if 'root_cause' in metadata:
            parts.append(f"Root cause: {metadata['root_cause']}")

        return 'n'.join(parts) if parts else None
