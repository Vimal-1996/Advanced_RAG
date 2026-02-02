# src/storage.py
from sqlalchemy import create_engine, Column, Integer, String, Text, JSON, DateTime, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from typing import List, Dict, Any
from config.settings import settings
import logging

logger = logging.getLogger(__name__)

Base = declarative_base()


class DocumentChunk(Base):
    """Database model for document chunks"""
    __tablename__ = 'document_chunks'

    id = Column(Integer, primary_key=True, autoincrement=True)
    chunk_id = Column(String(255), unique=True, index=True, nullable=False)
    text = Column(Text, nullable=False)
    token_count = Column(Integer)
    char_count = Column(Integer)
    word_count = Column(Integer)
    source_page = Column(Integer, index=True)
    chunk_metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Create composite index for common queries
    __table_args__ = (
        Index('idx_page_created', 'source_page', 'created_at'),
    )

    def to_dict(self):
        return {
            'id': self.id,
            'chunk_id': self.chunk_id,
            'text': self.text,
            'token_count': self.token_count,
            'char_count': self.char_count,
            'word_count': self.word_count,
            'source_page': self.source_page,
            'metadata': self.chunk_metadata,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class ChunkStorage:
    """Handle database operations for document chunks"""

    def __init__(self, database_url: str = None):
        self.database_url = database_url or settings.get_database_url()

        # Create engine with connection pooling
        self.engine = create_engine(
            self.database_url,
            echo=False,
            pool_pre_ping=True,  # Verify connections before using
            pool_size=10,
            max_overflow=20
        )

        # Create all tables
        Base.metadata.create_all(self.engine)

        # Create session factory
        self.SessionLocal = sessionmaker(bind=self.engine)

        logger.info(f"✅ Database initialized: {self._mask_password(self.database_url)}")

    def _mask_password(self, url: str) -> str:
        """Mask password in database URL for logging"""
        import re
        return re.sub(r':([^:@]+)@', ':****@', url)

    def store_chunks(self, chunks: List[Dict[str, Any]], batch_size: int = 100) -> int:
        """Store chunks in database with batch processing"""
        session = self.SessionLocal()
        stored_count = 0

        try:
            logger.info(f"Storing {len(chunks):,} chunks in batches of {batch_size}...")

            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]

                for chunk in batch:
                    db_chunk = DocumentChunk(
                        chunk_id=chunk['chunk_id'],
                        text=chunk['text'],
                        token_count=chunk['token_count'],
                        char_count=chunk['char_count'],
                        word_count=chunk.get('word_count', 0),
                        source_page=chunk['source_page'],
                        chunk_metadata=chunk['metadata']
                    )
                    session.add(db_chunk)
                    stored_count += 1

                # Commit each batch
                session.commit()
                logger.info(f"Stored {stored_count:,}/{len(chunks):,} chunks ({stored_count / len(chunks) * 100:.1f}%)")

            logger.info(f"✅ Successfully stored {stored_count:,} chunks")
            return stored_count

        except Exception as e:
            session.rollback()
            logger.error(f"❌ Error storing chunks: {e}")
            raise
        finally:
            session.close()

    def get_all_chunks(self, limit: int = None) -> List[Dict[str, Any]]:
        """Retrieve all chunks (with optional limit)"""
        session = self.SessionLocal()
        try:
            query = session.query(DocumentChunk)
            if limit:
                query = query.limit(limit)
            chunks = query.all()
            return [chunk.to_dict() for chunk in chunks]
        finally:
            session.close()

    def get_chunk_by_id(self, chunk_id: str) -> Dict[str, Any]:
        """Get a specific chunk by ID"""
        session = self.SessionLocal()
        try:
            chunk = session.query(DocumentChunk).filter_by(chunk_id=chunk_id).first()
            return chunk.to_dict() if chunk else None
        finally:
            session.close()

    def get_chunk_count(self) -> int:
        """Get total chunk count"""
        session = self.SessionLocal()
        try:
            count = session.query(DocumentChunk).count()
            logger.info(f"Total chunks in database: {count:,}")
            return count
        finally:
            session.close()

    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        session = self.SessionLocal()
        try:
            from sqlalchemy import func

            stats = session.query(
                func.count(DocumentChunk.id).label('total_chunks'),
                func.sum(DocumentChunk.token_count).label('total_tokens'),
                func.avg(DocumentChunk.token_count).label('avg_tokens'),
                func.sum(DocumentChunk.char_count).label('total_chars'),
                func.count(func.distinct(DocumentChunk.source_page)).label('total_pages')
            ).first()

            return {
                'total_chunks': stats.total_chunks or 0,
                'total_tokens': stats.total_tokens or 0,
                'avg_tokens_per_chunk': float(stats.avg_tokens or 0),
                'total_characters': stats.total_chars or 0,
                'total_pages': stats.total_pages or 0
            }
        finally:
            session.close()

    def clear_all_chunks(self):
        """Clear all chunks (for reprocessing)"""
        session = self.SessionLocal()
        try:
            deleted_count = session.query(DocumentChunk).delete()
            session.commit()
            logger.info(f"✅ Cleared {deleted_count:,} chunks from database")
        except Exception as e:
            session.rollback()
            logger.error(f"❌ Error clearing chunks: {e}")
            raise
        finally:
            session.close()

    def test_connection(self) -> bool:
        """Test database connection"""
        try:
            from sqlalchemy import text  # ✅ Import text
            session = self.SessionLocal()
            session.execute(text("SELECT 1"))
            session.close()
            logger.info("✅ Database connection successful")
            return True
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            return False