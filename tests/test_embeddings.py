# tests/test_embeddings.py
from src.Embeddings.embedding_service import EmbeddingService
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_embedding_generation():
    """Test embedding generation"""

    logger.info("Testing embedding generation...")

    service = EmbeddingService()

    # Test single embedding
    text = "Machine learning is a subset of artificial intelligence."
    embedding = service.generate_embedding(text)

    assert len(embedding) == service.dimensions
    assert all(isinstance(x, float) for x in embedding)

    logger.info(f"✅ Single embedding: {len(embedding)} dimensions")

    # Test batch embeddings
    texts = [
        "Deep learning uses neural networks.",
        "AI is transforming technology.",
        "Natural language processing analyzes text."
    ]

    embeddings = service.generate_embeddings_batch(texts)

    assert len(embeddings) == len(texts)
    assert all(len(emb) == service.dimensions for emb in embeddings)

    logger.info(f"✅ Batch embeddings: {len(embeddings)} vectors")

    # Check usage stats
    stats = service.get_usage_stats()
    logger.info(f"✅ Usage: {stats['total_tokens']} tokens, ${stats['total_cost_usd']:.4f}")


if __name__ == "__main__":
    test_embedding_generation()
    logger.info("\n✅ All embedding tests passed!")