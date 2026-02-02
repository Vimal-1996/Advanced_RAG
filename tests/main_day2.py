# main_day2.py
from src.vector_strore.vector_manager import VectorManager
from src.retrieval.retriever import Retriever
from config.settings import settings
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_banner(message: str):
    """Print formatted banner"""
    logger.info("=" * 70)
    logger.info(f"  {message}")
    logger.info("=" * 70)


def main():
    """Day 2: Complete embedding and vector indexing pipeline"""

    try:
        print_banner("DAY 2: EMBEDDING GENERATION & VECTOR DATABASE")

        # Initialize vector manager
        vector_manager = VectorManager()

        # Step 1: Initialize vector database
        print_banner("STEP 1: Initialize Vector Database")

        recreate = input("Recreate collection? (y/N): ").strip().lower() == 'y'
        vector_manager.initialize_vector_database(recreate=recreate)

        # Step 2: Index all chunks
        print_banner("STEP 2: Generate Embeddings & Index Vectors")

        resume = input("Resume from checkpoint? (Y/n): ").strip().lower() != 'n'
        result = vector_manager.index_all_chunks(resume=resume)

        # Step 3: Test retrieval
        print_banner("STEP 3: Test Retrieval")

        retriever = Retriever()

        # Test queries
        test_queries = [
            "What is machine learning?",
            "How does deep learning work?",
            "What are neural networks?"
        ]

        for query in test_queries:
            logger.info(f"\n📝 Test Query: {query}")
            results = retriever.search(query, top_k=3)

            for idx, result in enumerate(results, 1):
                logger.info(f"\n   Result {idx}:")
                logger.info(f"   Score: {result['score']:.4f}")
                logger.info(f"   Page: {result['source_page']}")
                logger.info(f"   Text: {result['text'][:150]}...")

        # Step 4: Show final statistics
        print_banner("FINAL STATISTICS")

        stats = vector_manager.get_statistics()

        logger.info("📊 Database Statistics:")
        logger.info(f"   Total chunks: {stats['database']['total_chunks']:,}")
        logger.info(f"   Total tokens: {stats['database']['total_tokens']:,}")
        logger.info(f"   Total pages: {stats['database']['total_pages']:,}")

        logger.info("\n📊 Vector Store Statistics:")
        logger.info(f"   Collection: {stats['vector_store'].get('name')}")
        logger.info(f"   Vectors: {stats['vector_store'].get('vectors_count', 0):,}")
        logger.info(f"   Vector size: {stats['vector_store'].get('vector_size')}")

        logger.info("\n📊 Embedding Statistics:")
        logger.info(f"   Provider: {stats['embeddings']['provider']}")
        logger.info(f"   Model: {stats['embeddings']['model']}")
        logger.info(f"   Total tokens: {stats['embeddings']['total_tokens']:,}")
        logger.info(f"   Total cost: ${stats['embeddings']['total_cost_usd']:.4f}")

        print_banner("✅ DAY 2 COMPLETE")
        logger.info("Next steps:")
        logger.info("  - Day 3: Build API backend with hybrid search")
        logger.info("  - Day 4: Integrate LLM for answer generation")

        return True

    except KeyboardInterrupt:
        logger.warning("\n⚠️  Pipeline interrupted by user")
        return False
    except Exception as e:
        logger.error(f"\n❌ Pipeline failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)