# src/vector_store/qdrant_store.py
from typing import List, Dict, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    Range
)
from config.settings import settings
import logging

logger = logging.getLogger(__name__)


class QdrantStore:
    """Qdrant vector database operations"""

    def __init__(self):
        # Connect to Qdrant
        if settings.QDRANT_URL:
            # Qdrant Cloud
            self.client = QdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY
            )
            logger.info(f"✅ Connected to Qdrant Cloud: {settings.QDRANT_URL}")
        else:
            # Local Qdrant (Docker)
            self.client = QdrantClient(
                host=settings.QDRANT_HOST,
                port=settings.QDRANT_PORT,
                grpc_port=6334 if settings.QDRANT_USE_GRPC else None
            )
            logger.info(f"✅ Connected to Qdrant: {settings.QDRANT_HOST}:{settings.QDRANT_PORT}")

        self.collection_name = settings.QDRANT_COLLECTION_NAME
        self.vector_size = settings.OPENAI_EMBEDDING_DIMENSIONS

    def create_collection(self, recreate: bool = False):
        """Create vector collection"""

        # Check if collection exists
        collections = self.client.get_collections().collections
        collection_exists = any(c.name == self.collection_name for c in collections)

        if collection_exists:
            if recreate:
                logger.warning(f"⚠️  Deleting existing collection: {self.collection_name}")
                self.client.delete_collection(self.collection_name)
            else:
                logger.info(f"✅ Collection already exists: {self.collection_name}")
                return

        # Create collection
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.vector_size,
                distance=Distance.COSINE
            )
        )

        logger.info(f"✅ Created collection: {self.collection_name}")
        logger.info(f"   Vector size: {self.vector_size}")
        logger.info(f"   Distance metric: COSINE")

    def upsert_embeddings(
            self,
            embeddings_data: List[Dict],
            batch_size: int = 100
    ) -> int:
        """
        Upload embeddings to Qdrant

        Args:
            embeddings_data: List of dicts with 'chunk_id', 'embedding', 'source_page', etc.
            batch_size: Number of vectors to upload per batch

        Returns:
            Number of vectors uploaded
        """

        total_uploaded = 0

        # Process in batches
        for i in range(0, len(embeddings_data), batch_size):
            batch = embeddings_data[i:i + batch_size]

            # Create points
            points = []
            for idx, item in enumerate(batch):
                point = PointStruct(
                    id=total_uploaded + idx,  # Sequential ID
                    vector=item['embedding'],
                    payload={
                        'chunk_id': item['chunk_id'],
                        'source_page': item.get('source_page'),
                        'token_count': item.get('token_count')
                    }
                )
                points.append(point)

            # Upload batch
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )

            total_uploaded += len(points)

            if (i + batch_size) % 1000 == 0:
                logger.info(f"   Uploaded {total_uploaded:,} vectors...")

        logger.info(f"✅ Uploaded {total_uploaded:,} vectors to Qdrant")
        return total_uploaded

    def search(
            self,
            query_vector: List[float],
            limit: int = 10,
            score_threshold: float = None,
            filter_conditions: Dict = None
    ) -> List[Dict]:
        """
        Search for similar vectors

        Args:
            query_vector: Query embedding
            limit: Number of results to return
            score_threshold: Minimum similarity score
            filter_conditions: Metadata filters (e.g., {'source_page': 142})

        Returns:
            List of search results with scores and metadata
        """

        # Build filter if provided
        query_filter = None
        if filter_conditions:
            conditions = []

            if 'source_page' in filter_conditions:
                conditions.append(
                    FieldCondition(
                        key="source_page",
                        match=MatchValue(value=filter_conditions['source_page'])
                    )
                )

            if 'page_range' in filter_conditions:
                min_page, max_page = filter_conditions['page_range']
                conditions.append(
                    FieldCondition(
                        key="source_page",
                        range=Range(gte=min_page, lte=max_page)
                    )
                )

            if conditions:
                query_filter = Filter(must=conditions)

        # Search
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit,
            score_threshold=score_threshold,
            query_filter=query_filter
        )

        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                'chunk_id': result.payload['chunk_id'],
                'score': result.score,
                'source_page': result.payload.get('source_page'),
                'token_count': result.payload.get('token_count')
            })

        return formatted_results

    def get_collection_info(self) -> Dict:
        """Get collection statistics"""

        try:
            info = self.client.get_collection(self.collection_name)

            return {
                'name': self.collection_name,
                'vectors_count': info.vectors_count,
                'points_count': info.points_count,
                'status': info.status,
                'vector_size': self.vector_size
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {}

    def delete_collection(self):
        """Delete the collection"""
        self.client.delete_collection(self.collection_name)
        logger.info(f"✅ Deleted collection: {self.collection_name}")