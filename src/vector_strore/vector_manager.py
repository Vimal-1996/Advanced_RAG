# src/vector_store/vector_manager.py
from typing import List, Dict
from src.vector_strore.qdrant_store import QdrantStore
from src.Embeddings.embedding_service import EmbeddingService
from src.Embeddings.batch_processor import BatchEmbeddingProcessor
from src.storage import ChunkStorage
import logging

logger = logging.getLogger(__name__)


class VectorManager:
    """High-level vector operations manager"""

    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.vector_store = QdrantStore()
        self.storage = ChunkStorage()
        self.batch_processor = BatchEmbeddingProcessor(
            self.embedding_service,
            self.storage
        )

    def initialize_vector_database(self, recreate: bool = False):
        """Initialize vector database (create collection)"""
        logger.info("Initializing vector database...")
        self.vector_store.create_collection(recreate=recreate)

    def index_all_chunks(self, resume: bool = True) -> Dict:
        """
        Complete indexing pipeline:
        1. Generate embeddings for all chunks
        2. Upload to vector database

        Args:
            resume: Resume from checkpoint if available

        Returns:
            Statistics dictionary
        """

        # Step 1: Generate embeddings
        logger.info("=" * 60)
        logger.info("STEP 1: Generating Embeddings")
        logger.info("=" * 60)

        result = self.batch_processor.process_all_chunks(resume=resume)

        # ✅ FIX: Check if 'embeddings' key exists
        if 'embeddings' not in result:
            raise KeyError(
                "The batch processor did not return 'embeddings' key. "
                "Check batch_processor.py return statement."
            )

        embeddings_data = result['embeddings']

        if not embeddings_data:
            logger.warning("⚠️  No new embeddings to upload")

            # Get collection info anyway
            collection_info = self.vector_store.get_collection_info()

            return {
                **result,
                'uploaded_count': 0,
                'collection_info': collection_info
            }

        # Step 2: Upload to vector database
        logger.info("=" * 60)
        logger.info("STEP 2: Uploading to Vector Database")
        logger.info("=" * 60)

        uploaded_count = self.vector_store.upsert_embeddings(embeddings_data)

        # Get final stats
        collection_info = self.vector_store.get_collection_info()

        logger.info("=" * 60)
        logger.info("✅ INDEXING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Collection: {collection_info.get('name')}")
        logger.info(f"Total vectors: {collection_info.get('vectors_count', 0):,}")
        logger.info(f"Embedding cost: ${result['stats']['total_cost_usd']:.4f}")
        logger.info("=" * 60)

        return {
            **result,
            'uploaded_count': uploaded_count,
            'collection_info': collection_info
        }

    def get_statistics(self) -> Dict:
        """Get complete system statistics"""

        # Database stats
        db_stats = self.storage.get_statistics()

        # Vector store stats
        vector_stats = self.vector_store.get_collection_info()

        # Embedding stats
        embedding_stats = self.embedding_service.get_usage_stats()

        return {
            'database': db_stats,
            'vector_store': vector_stats,
            'embeddings': embedding_stats
        }