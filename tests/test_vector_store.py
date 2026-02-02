# tests/test_vector_store.py
from src.vector_store.qdrant_store import QdrantStore
from src.embeddings.embedding_service import EmbeddingService
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_vector_store():
    """Test Qdrant vector store operations"""

    logger.info("Testing Qdrant vector store...")

    # Initialize
    store = QdrantStore()
    embedding_service = EmbeddingService()

    # Create test collection
    test_collection = "test_collection"
    store.collection_name = test_collection
    store.create_collection(recreate=True)

    logger.info(f"✅ Created test collection: {test_collection}")

    # Create test data
    test_data = [
        {
            'chunk_id': 'test_1',
            'embedding': embedding_service.generate_embedding("Machine learning test"),
            'source_page': 1
        },
        {
            'chunk_id': 'test_2',
            'embedding': embedding_service.generate_embedding("Deep learning test"),
            'source_page': 2
        }
    ]

    # Upload
    count = store.upsert_embeddings(test_data)
    assert count == 2
    logger.info(f"✅ Uploaded {count} test vectors")

    # Search
    query_emb = embedding_service.generate_embedding("machine learning")
    results = store.search(query_emb, limit=2)

    assert len(results) > 0
    assert 'chunk_id' in results[0]
    assert 'score' in results[0]

    logger.info(f"✅ Search returned {len(results)} results")
    logger.info(f"   Top result: {results[0]['chunk_id']} (score: {results[0]['score']:.4f})")

    # Cleanup
    store.delete_collection()
    logger.info(f"✅ Deleted test collection")


if __name__ == "__main__":
    test_vector_store()
    logger.info("\n✅ All vector store tests passed!")