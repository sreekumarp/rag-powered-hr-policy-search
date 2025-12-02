"""
Unit tests for VectorStore module.
"""

import pytest
import tempfile
import shutil
import os
from utils.vector_store import VectorStore


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    temp = tempfile.mkdtemp()
    yield temp
    shutil.rmtree(temp)


@pytest.fixture
def vector_store(temp_dir):
    """Create VectorStore instance with temporary storage."""
    index_path = os.path.join(temp_dir, "test_index")
    vs = VectorStore(
        model_name="all-MiniLM-L6-v2",
        index_path=index_path,
        chunk_size=200,
        chunk_overlap=20,
    )
    return vs


class TestVectorStore:
    """Tests for VectorStore class."""

    def test_initialization(self, vector_store):
        """Test VectorStore initializes correctly."""
        assert vector_store.embedding_model is not None
        assert vector_store.embedding_dim == 384  # MiniLM dimension
        assert vector_store.index is None
        assert len(vector_store.documents) == 0

    def test_chunk_text_single_paragraph(self, vector_store):
        """Test chunking a single paragraph."""
        text = "This is a simple test paragraph."
        chunks = vector_store._chunk_text(text, "test_source")

        assert len(chunks) == 1
        assert chunks[0]["text"] == text
        assert chunks[0]["source"] == "test_source"
        assert chunks[0]["chunk_id"] == 0
        assert "char_count" in chunks[0]

    def test_chunk_text_multiple_paragraphs(self, vector_store):
        """Test chunking multiple paragraphs."""
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunks = vector_store._chunk_text(text, "test")

        assert len(chunks) >= 1
        assert all("source" in chunk for chunk in chunks)
        assert all("chunk_id" in chunk for chunk in chunks)

    def test_chunk_text_long_text(self, vector_store):
        """Test chunking text that exceeds chunk size."""
        # Create long text that should be split
        long_text = "This is a test sentence. " * 50
        chunks = vector_store._chunk_text(long_text, "long_test")

        assert len(chunks) >= 2  # Should be split into multiple chunks
        for i, chunk in enumerate(chunks):
            assert len(chunk["text"]) <= vector_store.chunk_size + 100  # Allow some flexibility

    def test_add_single_document(self, vector_store):
        """Test adding a single document."""
        text = "This is a test document about HR policies."
        result = vector_store.add_document(text, "doc1", {"category": "test"})

        assert result["chunks_added"] > 0
        assert result["source"] == "doc1"
        assert vector_store.index is not None
        assert len(vector_store.documents) > 0
        assert "doc1" in vector_store.document_metadata

    def test_add_multiple_documents(self, vector_store):
        """Test adding multiple documents."""
        texts = [
            "Document one about leave policies.",
            "Document two about remote work.",
            "Document three about benefits.",
        ]
        sources = ["doc1", "doc2", "doc3"]

        total_chunks = vector_store.add_documents(texts, sources)

        assert total_chunks > 0
        assert len(vector_store.document_metadata) == 3
        assert vector_store.index.ntotal == total_chunks

    def test_search_basic(self, vector_store):
        """Test basic search functionality."""
        # Add test documents
        vector_store.add_document(
            "Annual leave policy: Employees get 20 days of vacation.",
            "leave_doc",
            {"category": "leave"},
        )
        vector_store.add_document(
            "Remote work policy: Work from home twice per week.",
            "remote_doc",
            {"category": "remote"},
        )

        # Search
        results = vector_store.search("How many vacation days?", k=2)

        assert len(results) > 0
        assert len(results) <= 2
        # First result should be about leave (higher relevance)
        doc, score = results[0]
        assert "leave" in doc["text"].lower() or "vacation" in doc["text"].lower()
        assert 0 <= score <= 1

    def test_search_with_filter(self, vector_store):
        """Test search with metadata filtering."""
        # Add documents with different categories
        vector_store.add_document(
            "Leave policy information.",
            "leave_doc",
            {"category": "leave"},
        )
        vector_store.add_document(
            "Benefits policy information.",
            "benefits_doc",
            {"category": "benefits"},
        )

        # Search with filter
        results = vector_store.search(
            "policy information", k=5, filter_metadata={"category": "leave"}
        )

        # Should only return leave documents
        assert len(results) > 0
        for doc, score in results:
            assert doc["category"] == "leave"

    def test_search_empty_store(self, vector_store):
        """Test search on empty vector store."""
        results = vector_store.search("test query", k=3)
        assert results == []

    def test_delete_document(self, vector_store):
        """Test document deletion."""
        # Add documents
        vector_store.add_document("Doc 1 content", "doc1")
        vector_store.add_document("Doc 2 content", "doc2")

        initial_count = len(vector_store.documents)

        # Delete one
        success = vector_store.delete_document("doc1")

        assert success is True
        assert "doc1" not in vector_store.document_metadata
        assert len(vector_store.documents) < initial_count

    def test_delete_nonexistent_document(self, vector_store):
        """Test deleting a document that doesn't exist."""
        success = vector_store.delete_document("nonexistent")
        assert success is False

    def test_get_document_list(self, vector_store):
        """Test getting list of indexed documents."""
        vector_store.add_document("Doc 1", "doc1", {"category": "a"})
        vector_store.add_document("Doc 2", "doc2", {"category": "b"})

        doc_list = vector_store.get_document_list()

        assert len(doc_list) == 2
        sources = [d["source"] for d in doc_list]
        assert "doc1" in sources
        assert "doc2" in sources

    def test_get_stats(self, vector_store):
        """Test statistics retrieval."""
        vector_store.add_document("Test document content", "test_doc")

        stats = vector_store.get_stats()

        assert "total_chunks" in stats
        assert "total_documents" in stats
        assert "embedding_model" in stats
        assert "embedding_dimension" in stats
        assert stats["total_documents"] == 1

    def test_persistence(self, temp_dir):
        """Test that index persists to disk."""
        index_path = os.path.join(temp_dir, "persist_test")

        # Create and populate store
        vs1 = VectorStore(index_path=index_path, chunk_size=200)
        vs1.add_document("Persistent document", "persist_doc", {"test": "data"})

        # Verify files exist
        assert os.path.exists(f"{index_path}.faiss")
        assert os.path.exists(f"{index_path}_docs.pkl")
        assert os.path.exists(f"{index_path}_metadata.json")

        # Create new instance (should load from disk)
        vs2 = VectorStore(index_path=index_path, chunk_size=200)

        assert len(vs2.documents) == len(vs1.documents)
        assert "persist_doc" in vs2.document_metadata

    def test_clear(self, vector_store):
        """Test clearing the vector store."""
        vector_store.add_document("Test", "test")
        vector_store.clear()

        assert vector_store.index is None
        assert len(vector_store.documents) == 0
        assert len(vector_store.document_metadata) == 0

    def test_matches_filter_exact(self, vector_store):
        """Test metadata filtering with exact match."""
        doc = {"category": "leave", "department": "hr"}

        assert vector_store._matches_filter(doc, {"category": "leave"}) is True
        assert vector_store._matches_filter(doc, {"category": "benefits"}) is False
        assert (
            vector_store._matches_filter(
                doc, {"category": "leave", "department": "hr"}
            )
            is True
        )

    def test_matches_filter_list(self, vector_store):
        """Test metadata filtering with list values."""
        doc = {"category": "leave"}

        assert (
            vector_store._matches_filter(doc, {"category": ["leave", "benefits"]})
            is True
        )
        assert (
            vector_store._matches_filter(doc, {"category": ["remote", "benefits"]})
            is False
        )
