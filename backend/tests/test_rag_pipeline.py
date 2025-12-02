"""
Unit tests for RAGPipeline module.
"""

import pytest
import tempfile
import shutil
import os
from utils.vector_store import VectorStore
from utils.rag_pipeline import RAGPipeline


@pytest.fixture
def temp_dir():
    """Create temporary directory."""
    temp = tempfile.mkdtemp()
    yield temp
    shutil.rmtree(temp)


@pytest.fixture
def rag_pipeline(temp_dir):
    """Create RAGPipeline instance."""
    index_path = os.path.join(temp_dir, "test_index")
    vs = VectorStore(
        model_name="all-MiniLM-L6-v2",
        index_path=index_path,
        chunk_size=200,
        chunk_overlap=20,
    )
    pipeline = RAGPipeline(
        vector_store=vs,
        retrieval_k=3,
        confidence_threshold=0.3,
        min_confidence_for_answer=0.5,
    )
    return pipeline


@pytest.fixture
def populated_pipeline(rag_pipeline):
    """Create pipeline with sample documents."""
    # Add sample documents
    rag_pipeline.vector_store.add_document(
        "Q: How many vacation days do I get?\nA: Full-time employees receive 20 days of paid annual leave per year.",
        "leave_policy",
        {"category": "leave_policies"},
    )
    rag_pipeline.vector_store.add_document(
        "Q: Can I work from home?\nA: Employees can work remotely up to 2 days per week after 6 months tenure.",
        "remote_work",
        {"category": "remote_work"},
    )
    rag_pipeline.vector_store.add_document(
        "Q: What are the sick leave rules?\nA: You get 10 paid sick days per year. Medical certificate required after 3 days.",
        "sick_leave",
        {"category": "leave_policies"},
    )
    return rag_pipeline


class TestRAGPipeline:
    """Tests for RAGPipeline class."""

    def test_initialization(self, rag_pipeline):
        """Test pipeline initializes correctly."""
        assert rag_pipeline.vector_store is not None
        assert rag_pipeline.retrieval_k == 3
        assert rag_pipeline.confidence_threshold == 0.3
        assert rag_pipeline.min_confidence_for_answer == 0.5
        assert rag_pipeline.query_count == 0

    def test_query_with_empty_store(self, rag_pipeline):
        """Test query with no documents indexed."""
        result = rag_pipeline.query("How many vacation days?")

        assert "answer" in result
        assert "sources" in result
        assert result["sources"] == []
        assert result["confidence"] == 0.0
        assert result["relevant_chunks"] == 0

    def test_query_basic(self, populated_pipeline):
        """Test basic query functionality."""
        result = populated_pipeline.query("How many vacation days do I get?")

        assert "answer" in result
        assert "sources" in result
        assert "confidence" in result
        assert "relevant_chunks" in result
        assert "query" in result
        assert "processing_time_ms" in result
        assert "timestamp" in result
        assert "warnings" in result

        assert len(result["sources"]) > 0
        assert result["confidence"] > 0

    def test_query_finds_relevant_content(self, populated_pipeline):
        """Test that query returns relevant content."""
        result = populated_pipeline.query("vacation days")

        # Should find vacation/leave information
        assert result["relevant_chunks"] > 0
        found_relevant = any(
            "vacation" in str(src).lower() or "leave" in str(src).lower()
            for src in result["sources"]
        )
        assert found_relevant

    def test_query_with_filter(self, populated_pipeline):
        """Test query with metadata filtering."""
        result = populated_pipeline.query(
            "days off", filters={"category": "leave_policies"}
        )

        # All sources should have the filtered category
        for source in result["sources"]:
            assert source.get("category") == "leave_policies"

    def test_query_statistics_updated(self, populated_pipeline):
        """Test that query statistics are updated."""
        initial_count = populated_pipeline.query_count

        populated_pipeline.query("test question")

        assert populated_pipeline.query_count == initial_count + 1
        assert populated_pipeline.total_latency_ms > 0

    def test_query_response_structure(self, populated_pipeline):
        """Test response has all required fields."""
        result = populated_pipeline.query("vacation")

        required_fields = [
            "answer",
            "sources",
            "confidence",
            "relevant_chunks",
            "query",
            "filters_applied",
            "processing_time_ms",
            "timestamp",
            "warnings",
        ]

        for field in required_fields:
            assert field in result

    def test_source_structure(self, populated_pipeline):
        """Test that sources have correct structure."""
        result = populated_pipeline.query("vacation days")

        if result["sources"]:
            source = result["sources"][0]
            assert "text" in source
            assert "source" in source
            assert "chunk_id" in source
            assert "relevance_score" in source

    def test_low_confidence_response(self, populated_pipeline):
        """Test response when confidence is low."""
        # Query something not in the documents
        result = populated_pipeline.query("quantum physics theories")

        # Should return but with warnings or low confidence
        assert "warnings" in result
        assert result["confidence"] <= populated_pipeline.confidence_threshold or len(
            result["warnings"]
        ) > 0

    def test_include_low_confidence_flag(self, populated_pipeline):
        """Test include_low_confidence parameter."""
        result = populated_pipeline.query(
            "something unrelated", include_low_confidence=True
        )

        # Should still return some result even if low confidence
        assert "answer" in result

    def test_generate_response_with_qa_format(self, populated_pipeline):
        """Test response generation extracts Q&A pairs."""
        result = populated_pipeline.query("vacation days")

        # Should format answer nicely
        assert "Based on" in result["answer"] or len(result["answer"]) > 0

    def test_categorize_section(self, rag_pipeline):
        """Test section categorization."""
        assert rag_pipeline._categorize_section("Annual Leave Policy") == "leave_policies"
        assert rag_pipeline._categorize_section("Sick Leave") == "sick_leave"
        assert rag_pipeline._categorize_section("Remote Work Policy") == "remote_work"
        assert (
            rag_pipeline._categorize_section("Benefits and Compensation") == "benefits"
        )
        assert rag_pipeline._categorize_section("Workplace Conduct") == "workplace_conduct"
        assert rag_pipeline._categorize_section("Parental Leave") == "parental_leave"
        assert (
            rag_pipeline._categorize_section("Training and Development") == "training"
        )
        assert rag_pipeline._categorize_section("Random Topic") == "general"

    def test_get_pipeline_info(self, populated_pipeline):
        """Test getting pipeline information."""
        # Make some queries first
        populated_pipeline.query("test")
        populated_pipeline.query("another test")

        info = populated_pipeline.get_pipeline_info()

        assert "configuration" in info
        assert "statistics" in info
        assert "vector_store" in info

        assert info["statistics"]["total_queries"] == 2
        assert info["statistics"]["average_latency_ms"] > 0

    def test_reset_statistics(self, populated_pipeline):
        """Test resetting statistics."""
        # Make queries
        populated_pipeline.query("test")
        populated_pipeline.query("test")

        # Reset
        populated_pipeline.reset_statistics()

        assert populated_pipeline.query_count == 0
        assert populated_pipeline.total_latency_ms == 0
        assert populated_pipeline.average_confidence == 0

    def test_empty_response(self, rag_pipeline):
        """Test empty response generation."""
        result = rag_pipeline._empty_response("test", 0, {"category": "test"})

        assert "answer" in result
        assert "sources" in result
        assert result["sources"] == []
        assert result["confidence"] == 0.0
        assert "warnings" in result

    def test_error_response(self, rag_pipeline):
        """Test error response generation."""
        result = rag_pipeline._error_response("Test error", "question", 0)

        assert "error" in result["answer"].lower()
        assert result["sources"] == []
        assert result["confidence"] == 0.0

    def test_multiple_queries_track_stats(self, populated_pipeline):
        """Test that multiple queries accumulate statistics correctly."""
        for i in range(5):
            populated_pipeline.query("vacation")

        info = populated_pipeline.get_pipeline_info()
        assert info["statistics"]["total_queries"] == 5

    def test_confidence_score_range(self, populated_pipeline):
        """Test that confidence scores are in valid range."""
        result = populated_pipeline.query("vacation days")

        assert 0 <= result["confidence"] <= 1

        for source in result["sources"]:
            assert 0 <= source["relevance_score"] <= 1

    def test_processing_time_positive(self, populated_pipeline):
        """Test that processing time is recorded."""
        result = populated_pipeline.query("vacation")

        assert result["processing_time_ms"] > 0

    def test_timestamp_present(self, populated_pipeline):
        """Test that timestamp is included."""
        result = populated_pipeline.query("vacation")

        assert "timestamp" in result
        assert "T" in result["timestamp"]  # ISO format contains T
