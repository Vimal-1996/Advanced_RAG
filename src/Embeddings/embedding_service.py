# src/embeddings/embedding_service.py
from typing import List, Dict, Optional
import openai
from tenacity import retry, stop_after_attempt, wait_exponential
from config.settings import settings
import logging
import time

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Generate embeddings using various providers"""

    def __init__(self, provider: str = None):
        self.provider = provider or settings.EMBEDDING_PROVIDER

        # ✅ Better error message
        if not self.provider:
            raise ValueError(
                "Embedding provider not configured!\n"
                "Please set EMBEDDING_PROVIDER in your .env file.\n"
                "Example: EMBEDDING_PROVIDER=openai"
            )

        if self.provider == "openai":
            # ✅ Check API key
            if not settings.OPENAI_API_KEY:
                raise ValueError(
                    "OpenAI API key not found!\n"
                    "Please set OPENAI_API_KEY in your .env file.\n"
                    "Get your API key from: https://platform.openai.com/api-keys"
                )

            self.client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
            self.model = settings.OPENAI_EMBEDDING_MODEL
            self.dimensions = settings.OPENAI_EMBEDDING_DIMENSIONS
            logger.info(f"✅ Initialized OpenAI embeddings: {self.model}")

        self.total_tokens = 0
        self.total_cost = 0.0

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text"""

        if self.provider == "openai":
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )

            # Track usage
            self.total_tokens += response.usage.total_tokens

            # Calculate cost (text-embedding-3-small: $0.02 per 1M tokens)
            cost_per_token = 0.02 / 1_000_000
            self.total_cost += response.usage.total_tokens * cost_per_token

            return response.data[0].embedding

        elif self.provider == "cohere":
            response = self.client.embed(
                texts=[text],
                model=self.model,
                input_type="search_document"
            )
            return response.embeddings[0]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def generate_embeddings_batch(
            self,
            texts: List[str],
            show_progress: bool = True
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts"""

        if not texts:
            return []

        if self.provider == "openai":
            response = self.client.embeddings.create(
                model=self.model,
                input=texts
            )

            # Track usage
            self.total_tokens += response.usage.total_tokens
            cost_per_token = 0.02 / 1_000_000
            self.total_cost += response.usage.total_tokens * cost_per_token

            return [item.embedding for item in response.data]

        elif self.provider == "cohere":
            response = self.client.embed(
                texts=texts,
                model=self.model,
                input_type="search_document"
            )
            return response.embeddings

    def get_usage_stats(self) -> Dict:
        """Get embedding usage statistics"""
        return {
            'provider': self.provider,
            'model': self.model,
            'total_tokens': self.total_tokens,
            'total_cost_usd': round(self.total_cost, 4),
            'dimensions': self.dimensions
        }