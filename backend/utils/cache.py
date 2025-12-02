"""
Semantic caching for LLM responses.

Critical for staying within GitHub Models free tier (50 req/day).
Uses embedding similarity to match semantically similar queries.
"""

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class ResponseCache:
    """
    Semantic cache for LLM responses.

    Features:
    - Semantic similarity matching (not exact string match)
    - TTL-based expiration
    - Persistence to disk
    - LRU eviction when full

    Critical for GitHub Models to stay within 50 requests/day limit.
    """

    def __init__(
        self,
        embedding_model,
        similarity_threshold: float = 0.92,
        max_size: int = 500,
        default_ttl: int = 86400,  # 24 hours
        cache_file: str = "data/llm_cache.json"
    ):
        """
        Initialize semantic cache.

        Args:
            embedding_model: SentenceTransformer model for embeddings
            similarity_threshold: Min similarity for cache hit (0.0-1.0)
            max_size: Maximum cache entries
            default_ttl: Default TTL in seconds (24 hours for GitHub Models)
            cache_file: Path to cache persistence file
        """
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache_file = cache_file

        # Cache storage
        self.entries: List[Dict] = []
        self.embeddings: Optional[np.ndarray] = None

        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0

        # Load from disk
        self._load_from_disk()

    def get(self, query: str) -> Optional[Dict]:
        """
        Get cached response for semantically similar query.

        Args:
            query: User query

        Returns:
            Cached response dict or None if no match
        """
        if not self.entries:
            self.misses += 1
            return None

        # Embed query
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)[0]

        # Calculate similarities with all cached queries
        similarities = np.dot(self.embeddings, query_embedding)

        # Find best match
        best_idx = int(np.argmax(similarities))
        best_similarity = float(similarities[best_idx])

        logger.debug(f"Cache lookup: best similarity = {best_similarity:.4f} (threshold = {self.similarity_threshold})")

        # Check if above threshold
        if best_similarity >= self.similarity_threshold:
            entry = self.entries[best_idx]

            # Check if expired
            if self._is_expired(entry):
                logger.info(f"Cache entry expired, removing")
                self._remove_entry(best_idx)
                self.misses += 1
                return None

            # Cache hit!
            self.hits += 1
            entry["last_accessed"] = datetime.utcnow().isoformat()
            entry["access_count"] += 1

            logger.info(f"Cache HIT! Similarity: {best_similarity:.4f}, saved API call")

            return {
                "answer": entry["response"],
                "metadata": {
                    **entry.get("metadata", {}),
                    "cached": True,
                    "cache_similarity": round(best_similarity, 4),
                    "original_query": entry["query"],
                }
            }

        # No match
        self.misses += 1
        return None

    def set(
        self,
        query: str,
        response: str,
        metadata: Dict = None,
        ttl: int = None
    ) -> None:
        """
        Add entry to cache.

        Args:
            query: User query
            response: LLM response
            metadata: Optional metadata
            ttl: TTL in seconds (default: 24 hours)
        """
        ttl = ttl if ttl is not None else self.default_ttl

        # Embed query
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)[0]

        # Create entry
        entry = {
            "query": query,
            "response": response,
            "metadata": metadata or {},
            "created_at": datetime.utcnow().isoformat(),
            "last_accessed": datetime.utcnow().isoformat(),
            "expires_at": (datetime.utcnow() + timedelta(seconds=ttl)).isoformat(),
            "access_count": 0,
        }

        # Add to cache
        self.entries.append(entry)

        # Update embeddings matrix
        if self.embeddings is None:
            self.embeddings = query_embedding.reshape(1, -1)
        else:
            self.embeddings = np.vstack([self.embeddings, query_embedding])

        logger.info(f"Cached new response (cache size: {len(self.entries)}/{self.max_size})")

        # Evict if over capacity
        if len(self.entries) > self.max_size:
            self._evict_lru()

        # Persist to disk
        self._save_to_disk()

    def _is_expired(self, entry: Dict) -> bool:
        """Check if cache entry is expired."""
        expires_at = datetime.fromisoformat(entry["expires_at"])
        return datetime.utcnow() > expires_at

    def _remove_entry(self, idx: int) -> None:
        """Remove cache entry at index."""
        del self.entries[idx]
        self.embeddings = np.delete(self.embeddings, idx, axis=0)

        if len(self.entries) == 0:
            self.embeddings = None

        self._save_to_disk()

    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        # Find LRU (lowest access count, oldest access time)
        lru_idx = min(
            range(len(self.entries)),
            key=lambda i: (
                self.entries[i]["access_count"],
                self.entries[i]["last_accessed"]
            )
        )

        logger.info(f"Evicting LRU cache entry (index {lru_idx})")
        self._remove_entry(lru_idx)
        self.evictions += 1

    def clear(self) -> None:
        """Clear all cache entries."""
        self.entries = []
        self.embeddings = None
        logger.info("Cache cleared")
        self._save_to_disk()

    def get_statistics(self) -> Dict:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0

        return {
            "enabled": True,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(hit_rate, 4),
            "size": len(self.entries),
            "max_size": self.max_size,
            "evictions": self.evictions,
            "similarity_threshold": self.similarity_threshold,
        }

    def _save_to_disk(self) -> None:
        """Persist cache to disk (without embeddings)."""
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)

            with open(self.cache_file, 'w') as f:
                json.dump({
                    "entries": self.entries,
                    "statistics": {
                        "hits": self.hits,
                        "misses": self.misses,
                        "evictions": self.evictions
                    }
                }, f, indent=2)

            logger.debug(f"Cache persisted to {self.cache_file}")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    def _load_from_disk(self) -> None:
        """Load cache from disk and rebuild embeddings."""
        try:
            if not os.path.exists(self.cache_file):
                logger.info("No cache file found, starting fresh")
                return

            with open(self.cache_file, 'r') as f:
                data = json.load(f)

            # Load entries
            self.entries = data.get("entries", [])

            # Restore statistics
            stats = data.get("statistics", {})
            self.hits = stats.get("hits", 0)
            self.misses = stats.get("misses", 0)
            self.evictions = stats.get("evictions", 0)

            # Rebuild embeddings from queries
            if self.entries:
                queries = [entry["query"] for entry in self.entries]
                self.embeddings = self.embedding_model.encode(queries, convert_to_numpy=True)
                logger.info(f"Loaded {len(self.entries)} cache entries from disk")

        except Exception as e:
            logger.error(f"Failed to load cache: {e}, starting fresh")
            self.entries = []
            self.embeddings = None
