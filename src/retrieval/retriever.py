# src/retrieval/retriever.py
from typing import List, Dict, Optional
from src.vector_strore.qdrant_store import QdrantStore
from src.Embeddings.embedding_service import EmbeddingService
from src.storage import ChunkStorage
import logging
import time

logger = logging.getLogger(__name__)


class Retriever:
    """Retrieve relevant chunks for a query"""

    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.vector_store = QdrantStore()
        self.storage = ChunkStorage()

    def search(
            self,
            query: str,
            top_k: int = 10,
            score_threshold: float = 0.7,
            filters: Dict = None
    ) -> List[Dict]:
        """
        Search for relevant chunks

        Args:
            query: Search query
            top_k: Number of results to return
            score_threshold: Minimum similarity score
            filters: Metadata filters (e.g., {'source_page': 142})

        Returns:
            List of chunks with scores
        """

        start_time = time.time()

        # Step 1: Generate query embedding
        logger.info(f"🔍 Query: {query}")
        query_embedding = self.embedding_service.generate_embedding(query)

        # Step 2: Search vector database
        vector_results = self.vector_store.search(
            query_vector=query_embedding,
            limit=top_k,
            score_threshold=score_threshold,
            filter_conditions=filters
        )

        if not vector_results:
            logger.warning("No results found")
            return []

        # Step 3: Get full chunk data from PostgreSQL
        chunk_ids = [result['chunk_id'] for result in vector_results]
        chunks_dict = {}

        for chunk_id in chunk_ids:
            chunk = self.storage.get_chunk_by_id(chunk_id)
            if chunk:
                chunks_dict[chunk_id] = chunk

        # Step 4: Combine vector scores with chunk data
        results = []
        for vector_result in vector_results:
            chunk_id = vector_result['chunk_id']
            if chunk_id in chunks_dict:
                chunk = chunks_dict[chunk_id]
                results.append({
                    **chunk,
                    'score': vector_result['score']
                })

        elapsed_time = (time.time() - start_time) * 1000

        logger.info(f"✅ Found {len(results)} results in {elapsed_time:.1f}ms")

        return results

    def search_by_chunk_id(self, chunk_id: str, top_k: int = 10) -> List[Dict]:
        """Find chunks similar to a given chunk"""

        # Get the chunk
        chunk = self.storage.get_chunk_by_id(chunk_id)
        if not chunk:
            logger.error(f"Chunk not found: {chunk_id}")
            return []

        # Search using the chunk's text
        return self.search(chunk['text'], top_k=top_k)