# src/chunker.py
import tiktoken
from typing import List, Dict, Any
from config.settings import settings
import logging
import re

logger = logging.getLogger(__name__)


class SmartChunker:
    """Intelligent text chunking with overlap and metadata preservation"""

    def __init__(self, chunk_size: int = None, overlap: int = None, model: str = "gpt-3.5-turbo"):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.overlap = overlap or settings.CHUNK_OVERLAP

        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            logger.warning(f"Model {model} not found, using cl100k_base encoding")
            self.encoding = tiktoken.get_encoding("cl100k_base")

        logger.info(f"Chunker initialized: chunk_size={self.chunk_size}, overlap={self.overlap}")

    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoding.encode(text))

    def split_text_by_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex"""
        # Split on sentence boundaries (., !, ?) followed by whitespace
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create overlapping chunks from text while preserving sentence boundaries"""

        sentences = self.split_text_by_sentences(text)
        chunks = []
        current_chunk = []
        current_tokens = 0
        chunk_id = 0

        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)

            # If single sentence exceeds chunk size, split it by words
            if sentence_tokens > self.chunk_size:
                # Save current chunk if exists
                if current_chunk:
                    chunks.append(self._create_chunk(
                        current_chunk,
                        chunk_id,
                        metadata,
                        current_tokens
                    ))
                    chunk_id += 1
                    current_chunk = []
                    current_tokens = 0

                # Split long sentence into smaller parts
                word_chunks = self._split_long_sentence(sentence, metadata, chunk_id)
                chunks.extend(word_chunks)
                chunk_id += len(word_chunks)
                continue

            # Normal sentence processing
            if current_tokens + sentence_tokens > self.chunk_size:
                # Save current chunk
                chunks.append(self._create_chunk(
                    current_chunk,
                    chunk_id,
                    metadata,
                    current_tokens
                ))
                chunk_id += 1

                # Start new chunk with overlap
                overlap_chunk = self._get_overlap_text(current_chunk, current_tokens)
                current_chunk = overlap_chunk + [sentence]
                current_tokens = self.count_tokens(" ".join(current_chunk))
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens

        # Add final chunk
        if current_chunk:
            chunks.append(self._create_chunk(
                current_chunk,
                chunk_id,
                metadata,
                current_tokens
            ))

        return chunks

    def _split_long_sentence(self, sentence: str, metadata: Dict[str, Any], start_chunk_id: int) -> List[
        Dict[str, Any]]:
        """Split a very long sentence into word-based chunks"""
        words = sentence.split()
        chunks = []
        temp_chunk = []
        temp_tokens = 0
        chunk_id = start_chunk_id

        for word in words:
            word_tokens = self.count_tokens(word + " ")

            if temp_tokens + word_tokens > self.chunk_size:
                if temp_chunk:
                    chunks.append(self._create_chunk(
                        [" ".join(temp_chunk)],
                        chunk_id,
                        metadata,
                        temp_tokens
                    ))
                    chunk_id += 1
                temp_chunk = [word]
                temp_tokens = word_tokens
            else:
                temp_chunk.append(word)
                temp_tokens += word_tokens

        if temp_chunk:
            chunks.append(self._create_chunk(
                [" ".join(temp_chunk)],
                chunk_id,
                metadata,
                temp_tokens
            ))

        return chunks

    def _get_overlap_text(self, sentences: List[str], total_tokens: int) -> List[str]:
        """Get sentences for overlap from end of previous chunk"""
        overlap_sentences = []
        overlap_tokens = 0

        for sentence in reversed(sentences):
            sentence_tokens = self.count_tokens(sentence)
            if overlap_tokens + sentence_tokens <= self.overlap:
                overlap_sentences.insert(0, sentence)
                overlap_tokens += sentence_tokens
            else:
                break

        return overlap_sentences

    def _create_chunk(
            self,
            sentences: List[str],
            chunk_id: int,
            metadata: Dict[str, Any],
            token_count: int
    ) -> Dict[str, Any]:
        """Create chunk dictionary with metadata"""
        text = " ".join(sentences)

        return {
            'chunk_id': f"{metadata.get('page_number', 0)}_{chunk_id}",
            'text': text,
            'token_count': token_count,
            'char_count': len(text),
            'word_count': len(text.split()),
            'source_page': metadata.get('page_number'),
            'metadata': metadata
        }

    def process_pages(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process multiple pages into chunks"""
        all_chunks = []

        logger.info(f"Chunking {len(pages):,} pages...")

        for idx, page in enumerate(pages, start=1):
            page_chunks = self.chunk_text(page['text'], {
                'page_number': page.get('page_number', page.get('paragraph_number')),
                'original_char_count': page['char_count'],
                'original_word_count': page.get('word_count', 0)
            })
            all_chunks.extend(page_chunks)

            if idx % 100 == 0:
                logger.info(f"Chunked {idx:,}/{len(pages):,} pages ({idx / len(pages) * 100:.1f}%)")

        total_tokens = sum(c['token_count'] for c in all_chunks)
        avg_tokens = total_tokens / len(all_chunks) if all_chunks else 0

        logger.info(f"✅ Created {len(all_chunks):,} chunks")
        logger.info(f"   Total tokens: {total_tokens:,}")
        logger.info(f"   Avg tokens/chunk: {avg_tokens:.1f}")

        return all_chunks