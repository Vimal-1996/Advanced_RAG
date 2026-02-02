# src/embeddings/batch_processor.py
from typing import List, Dict, Optional
from tqdm import tqdm
import time
import json
from pathlib import Path
from config.settings import settings
from src.storage import ChunkStorage
from src.Embeddings.embedding_service import EmbeddingService
import logging

logger = logging.getLogger(__name__)


class BatchEmbeddingProcessor:
    """Process chunks in batches to generate embeddings"""

    def __init__(
            self,
            embedding_service: EmbeddingService,
            storage: ChunkStorage,
            batch_size: int = None
    ):
        self.embedding_service = embedding_service
        self.storage = storage
        self.batch_size = batch_size or settings.EMBEDDING_BATCH_SIZE
        self.checkpoint_dir = Path(settings.CHECKPOINT_DIR)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.checkpoint_file = self.checkpoint_dir / "embedding_progress.json"

    def load_checkpoint(self) -> Dict:
        """Load progress from checkpoint file"""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                return json.load(f)
        return {'processed_chunk_ids': [], 'last_batch': 0}

    def save_checkpoint(self, processed_chunk_ids: List[str], batch_num: int):
        """Save progress to checkpoint file"""
        checkpoint = {
            'processed_chunk_ids': processed_chunk_ids,
            'last_batch': batch_num,
            'timestamp': time.time()
        }
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)

    def process_all_chunks(self, resume: bool = True) -> Dict:
        """
        Generate embeddings for all chunks in database

        Args:
            resume: If True, resume from last checkpoint

        Returns:
            Dictionary with processing statistics
        """

        # Load checkpoint if resuming
        checkpoint = self.load_checkpoint() if resume else {'processed_chunk_ids': [], 'last_batch': 0}
        processed_ids = set(checkpoint['processed_chunk_ids'])

        # Get all chunks from database
        logger.info("Loading chunks from database...")
        all_chunks = self.storage.get_all_chunks()

        # Filter out already processed chunks
        chunks_to_process = [
            chunk for chunk in all_chunks
            if chunk['chunk_id'] not in processed_ids
        ]

        if not chunks_to_process:
            logger.info("✅ All chunks already have embeddings!")
            return {
                'embeddings': [],  # ✅ Always return 'embeddings' key
                'total_chunks': len(all_chunks),
                'processed': len(all_chunks),
                'skipped': len(all_chunks),
                'embeddings_generated': 0,
                'stats': self.embedding_service.get_usage_stats(),
                'elapsed_time': 0
            }

        logger.info(f"📊 Total chunks: {len(all_chunks)}")
        logger.info(f"✅ Already processed: {len(processed_ids)}")
        logger.info(f"🔄 To process: {len(chunks_to_process)}")

        # Create batches
        batches = [
            chunks_to_process[i:i + self.batch_size]
            for i in range(0, len(chunks_to_process), self.batch_size)
        ]

        logger.info(f"Processing {len(batches)} batches of {self.batch_size} chunks...")

        embeddings_data = []
        start_time = time.time()

        # Process batches with progress bar
        for batch_idx, batch in enumerate(tqdm(batches, desc="Generating embeddings")):
            # Extract texts
            texts = [chunk['text'] for chunk in batch]
            chunk_ids = [chunk['chunk_id'] for chunk in batch]

            # Generate embeddings
            try:
                embeddings = self.embedding_service.generate_embeddings_batch(texts)

                # Combine with metadata
                for chunk, embedding in zip(batch, embeddings):
                    embeddings_data.append({
                        'chunk_id': chunk['chunk_id'],
                        'embedding': embedding,
                        'source_page': chunk['source_page'],
                        'token_count': chunk['token_count']
                    })

                # Update checkpoint
                processed_ids.update(chunk_ids)

                # Save checkpoint periodically
                if (batch_idx + 1) % 10 == 0:  # Every 10 batches
                    self.save_checkpoint(list(processed_ids), batch_idx + 1)
                    logger.info(f"📌 Checkpoint saved: {len(processed_ids)} chunks processed")

            except Exception as e:
                logger.error(f"❌ Error processing batch {batch_idx}: {e}")
                # Save checkpoint before raising
                self.save_checkpoint(list(processed_ids), batch_idx)
                raise

        # Final checkpoint
        self.save_checkpoint(list(processed_ids), len(batches))

        elapsed_time = time.time() - start_time

        # Get usage stats
        stats = self.embedding_service.get_usage_stats()

        logger.info("=" * 60)
        logger.info("✅ Embedding generation complete!")
        logger.info(f"   Processed: {len(embeddings_data)} chunks")
        logger.info(f"   Total tokens: {stats['total_tokens']:,}")
        logger.info(f"   Total cost: ${stats['total_cost_usd']:.4f}")
        logger.info(f"   Time: {elapsed_time:.2f}s ({elapsed_time / 60:.2f} min)")
        logger.info(f"   Speed: {len(embeddings_data) / elapsed_time:.1f} chunks/sec")
        logger.info("=" * 60)

        # ✅ FIX: Always return 'embeddings' key
        return {
            'embeddings': embeddings_data,  # ✅ KEY FIX
            'total_chunks': len(all_chunks),
            'processed': len(embeddings_data),
            'skipped': len(processed_ids) - len(embeddings_data),
            'embeddings_generated': len(embeddings_data),
            'stats': stats,
            'elapsed_time': elapsed_time
        }