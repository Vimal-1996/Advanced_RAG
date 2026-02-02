# config/settings.py
from pydantic_settings import BaseSettings
from typing import Optional
import logging


class Settings(BaseSettings):
    """Application settings with environment variable support"""

    # Environment
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"

    # AWS Configuration
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    AWS_REGION: str = "us-east-1"
    S3_BUCKET_NAME: str
    S3_DOCUMENT_KEY: str

    # Database Configuration
    DATABASE_URL: str
    PROD_DATABASE_URL: Optional[str] = None

    # Document Processing
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    BATCH_SIZE: int = 100
    MAX_WORKERS: int = 4

    EMBEDDING_PROVIDER: str = "openai"

    # OpenAI Embeddings
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    OPENAI_EMBEDDING_DIMENSIONS: int = 1536

    # Embedding Batch Processing
    EMBEDDING_BATCH_SIZE: int = 100  # Process 100 chunks at a time
    EMBEDDING_RATE_LIMIT: int = 3000  # Max requests per minute

    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION_NAME: str = "document_chunks"
    QDRANT_USE_GRPC: bool = False

    # Qdrant Cloud (optional - for production)
    QDRANT_URL: Optional[str] = None
    QDRANT_API_KEY: Optional[str] = None

    # Vector Search Configuration
    VECTOR_SEARCH_LIMIT: int = 10
    VECTOR_SEARCH_SCORE_THRESHOLD: float = 0.7

    CHECKPOINT_DIR: str = "checkpoints"
    CHECKPOINT_INTERVAL: int = 100


    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"  # Ignore extra fields in .env

    def get_database_url(self) -> str:
        """Get appropriate database URL based on environment"""
        if self.ENVIRONMENT == "production" and self.PROD_DATABASE_URL:
            return self.PROD_DATABASE_URL
        return self.DATABASE_URL

    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, self.LOG_LEVEL),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )


# Global settings instance
settings = Settings()
settings.setup_logging()