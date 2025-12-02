"""
Enhanced Vector Store Management using FAISS and Sentence Transformers

Phase 2 improvements:
- Document metadata storage
- Filtering by metadata
- Better error handling
- Statistics and analytics
"""

import os
import json
import pickle
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


class VectorStore:
    """Enhanced FAISS-based vector store with metadata support."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        index_path: str = "data/faiss_index",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        """Initialize the vector store."""
        self.model_name = model_name
        self.index_path = index_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialize embedding model
        print(f"Loading embedding model: {model_name}")
        self.embedding_model = SentenceTransformer(model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()

        # FAISS index and document storage
        self.index: Optional[faiss.Index] = None
        self.documents: List[Dict] = []

        # Document metadata index (for filtering)
        self.document_metadata: Dict[str, Dict] = {}

        # Load existing index
        self._load_index()

    def _chunk_text(
        self, text: str, source: str = "unknown", metadata: Dict = None
    ) -> List[Dict]:
        """Split text into overlapping chunks with metadata."""
        chunks = []
        metadata = metadata or {}
        paragraphs = text.split("\n\n")

        current_chunk = ""
        chunk_id = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(current_chunk) + len(para) + 2 > self.chunk_size and current_chunk:
                chunks.append(
                    {
                        "text": current_chunk.strip(),
                        "source": source,
                        "chunk_id": chunk_id,
                        "char_count": len(current_chunk.strip()),
                        "created_at": datetime.utcnow().isoformat(),
                        **metadata,
                    }
                )
                chunk_id += 1

                # Create overlap
                words = current_chunk.split()
                overlap_word_count = min(len(words), self.chunk_overlap // 5)
                overlap_text = " ".join(words[-overlap_word_count:]) if words else ""
                current_chunk = overlap_text + " " + para
            else:
                current_chunk += ("\n\n" if current_chunk else "") + para

        # Last chunk
        if current_chunk.strip():
            chunks.append(
                {
                    "text": current_chunk.strip(),
                    "source": source,
                    "chunk_id": chunk_id,
                    "char_count": len(current_chunk.strip()),
                    "created_at": datetime.utcnow().isoformat(),
                    **metadata,
                }
            )

        return chunks

    def add_document(
        self,
        text: str,
        source: str,
        metadata: Dict = None,
    ) -> Dict:
        """
        Add a single document to the vector store.

        Args:
            text: Document text content
            source: Source identifier
            metadata: Additional metadata (category, upload_date, etc.)

        Returns:
            Dictionary with indexing results
        """
        metadata = metadata or {}
        chunks = self._chunk_text(text, source, metadata)

        if not chunks:
            return {"chunks_added": 0, "source": source}

        # Generate embeddings
        texts_to_embed = [chunk["text"] for chunk in chunks]
        embeddings = self.embedding_model.encode(
            texts_to_embed, show_progress_bar=False, convert_to_numpy=True
        )

        # Normalize for cosine similarity
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings.astype("float32")

        # Create or update index
        if self.index is None:
            self.index = faiss.IndexFlatIP(self.embedding_dim)

        # Store starting index for this document
        start_idx = len(self.documents)

        # Add to index
        self.index.add(embeddings)
        self.documents.extend(chunks)

        # Store document metadata
        self.document_metadata[source] = {
            "source": source,
            "chunk_count": len(chunks),
            "start_index": start_idx,
            "end_index": start_idx + len(chunks),
            "indexed_at": datetime.utcnow().isoformat(),
            **metadata,
        }

        # Save index
        self._save_index()

        return {
            "chunks_added": len(chunks),
            "source": source,
            "total_chunks": len(self.documents),
        }

    def add_documents(
        self, texts: List[str], sources: List[str] = None, metadata_list: List[Dict] = None
    ) -> int:
        """Add multiple documents to the vector store."""
        if sources is None:
            sources = [f"doc_{i}" for i in range(len(texts))]
        if metadata_list is None:
            metadata_list = [{} for _ in range(len(texts))]

        total_chunks = 0
        for text, source, metadata in zip(texts, sources, metadata_list):
            result = self.add_document(text, source, metadata)
            total_chunks += result["chunks_added"]

        return total_chunks

    def search(
        self,
        query: str,
        k: int = 3,
        filter_metadata: Dict = None,
    ) -> List[Tuple[Dict, float]]:
        """
        Search for similar documents with optional filtering.

        Args:
            query: Search query text
            k: Number of results to return
            filter_metadata: Optional metadata filters

        Returns:
            List of (document, similarity_score) tuples
        """
        if self.index is None or len(self.documents) == 0:
            return []

        # Embed query
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        query_embedding = query_embedding.astype("float32")

        # Search (get more than k to allow for filtering)
        search_k = min(k * 3, len(self.documents)) if filter_metadata else k
        scores, indices = self.index.search(query_embedding, search_k)

        # Build results with optional filtering
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.documents):
                doc = self.documents[idx]
                score = float(scores[0][i])

                # Apply metadata filters if specified
                if filter_metadata:
                    if not self._matches_filter(doc, filter_metadata):
                        continue

                results.append((doc, score))

                # Stop if we have enough results
                if len(results) >= k:
                    break

        return results

    def _matches_filter(self, doc: Dict, filters: Dict) -> bool:
        """Check if document matches metadata filters."""
        for key, value in filters.items():
            if key not in doc:
                return False

            doc_value = doc[key]

            # Handle list filters (any match)
            if isinstance(value, list):
                if doc_value not in value:
                    return False
            # Handle exact match
            elif doc_value != value:
                return False

        return True

    def delete_document(self, source: str) -> bool:
        """
        Delete a document from the vector store by source.

        Note: This is inefficient for FAISS (requires rebuilding index).
        In production, use a proper vector database.

        Args:
            source: Source identifier to delete

        Returns:
            True if deleted, False if not found
        """
        if source not in self.document_metadata:
            return False

        # Filter out documents with this source
        new_documents = [doc for doc in self.documents if doc.get("source") != source]

        if len(new_documents) == len(self.documents):
            return False

        # Rebuild index
        self.documents = []
        self.index = None
        del self.document_metadata[source]

        if new_documents:
            texts = [doc["text"] for doc in new_documents]
            embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings.astype("float32")

            self.index = faiss.IndexFlatIP(self.embedding_dim)
            self.index.add(embeddings)
            self.documents = new_documents

        self._save_index()
        return True

    def get_document_list(self) -> List[Dict]:
        """Get list of indexed documents with metadata."""
        return list(self.document_metadata.values())

    def get_stats(self) -> Dict:
        """Get detailed statistics about the vector store."""
        total_chars = sum(doc.get("char_count", 0) for doc in self.documents)

        return {
            "total_chunks": len(self.documents),
            "total_documents": len(self.document_metadata),
            "total_characters": total_chars,
            "embedding_model": self.model_name,
            "embedding_dimension": self.embedding_dim,
            "index_type": type(self.index).__name__ if self.index else None,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "documents": self.get_document_list(),
        }

    def _save_index(self) -> None:
        """Save index, documents, and metadata to disk."""
        if self.index is None:
            return

        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, f"{self.index_path}.faiss")

        # Save documents
        with open(f"{self.index_path}_docs.pkl", "wb") as f:
            pickle.dump(self.documents, f)

        # Save metadata
        with open(f"{self.index_path}_metadata.json", "w") as f:
            json.dump(self.document_metadata, f, indent=2)

        print(f"Saved index to {self.index_path}")

    def _load_index(self) -> bool:
        """Load index, documents, and metadata from disk."""
        faiss_file = f"{self.index_path}.faiss"
        docs_file = f"{self.index_path}_docs.pkl"
        metadata_file = f"{self.index_path}_metadata.json"

        if os.path.exists(faiss_file) and os.path.exists(docs_file):
            try:
                self.index = faiss.read_index(faiss_file)
                with open(docs_file, "rb") as f:
                    self.documents = pickle.load(f)

                # Load metadata if exists
                if os.path.exists(metadata_file):
                    with open(metadata_file, "r") as f:
                        self.document_metadata = json.load(f)
                else:
                    self.document_metadata = {}

                print(f"Loaded index with {len(self.documents)} chunks from {len(self.document_metadata)} documents")
                return True
            except Exception as e:
                print(f"Error loading index: {e}")
                self.index = None
                self.documents = []
                self.document_metadata = {}
                return False
        else:
            print("No existing index found")
            return False

    def clear(self) -> None:
        """Clear the vector store completely."""
        self.index = None
        self.documents = []
        self.document_metadata = {}

        # Remove files
        for ext in [".faiss", "_docs.pkl", "_metadata.json"]:
            file_path = f"{self.index_path}{ext}"
            if os.path.exists(file_path):
                os.remove(file_path)

        print("Vector store cleared")
