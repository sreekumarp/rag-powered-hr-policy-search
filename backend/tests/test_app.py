"""
Integration tests for Flask application.
"""

import pytest
import json
import tempfile
import shutil
import os
from app import app, vector_store, rag_pipeline


@pytest.fixture
def client():
    """Create test client."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def clean_store():
    """Ensure clean vector store for each test."""
    vector_store.clear()
    rag_pipeline.reset_statistics()
    yield
    vector_store.clear()


class TestAPIEndpoints:
    """Tests for Flask API endpoints."""

    def test_home_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        data = json.loads(response.data)

        assert response.status_code == 200
        assert "name" in data
        assert "version" in data
        assert "endpoints" in data

    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        data = json.loads(response.data)

        assert response.status_code == 200
        assert data["status"] == "healthy"
        assert "vector_store_ready" in data

    def test_stats_endpoint(self, client):
        """Test stats endpoint."""
        response = client.get("/stats")
        data = json.loads(response.data)

        assert response.status_code == 200
        assert "configuration" in data
        assert "statistics" in data
        assert "vector_store" in data

    def test_query_without_documents(self, client, clean_store):
        """Test query when no documents are indexed."""
        response = client.post(
            "/query",
            json={"question": "How many vacation days?"},
            content_type="application/json",
        )
        data = json.loads(response.data)

        assert response.status_code == 400
        assert "error" in data

    def test_query_missing_question(self, client):
        """Test query without question field."""
        response = client.post("/query", json={}, content_type="application/json")
        data = json.loads(response.data)

        assert response.status_code == 400
        assert "error" in data

    def test_query_empty_question(self, client):
        """Test query with empty question."""
        response = client.post(
            "/query", json={"question": ""}, content_type="application/json"
        )
        data = json.loads(response.data)

        assert response.status_code == 400

    def test_query_short_question(self, client):
        """Test query with too short question."""
        response = client.post(
            "/query", json={"question": "ab"}, content_type="application/json"
        )

        assert response.status_code == 400

    def test_query_long_question(self, client):
        """Test query with too long question."""
        long_question = "x" * 1001
        response = client.post(
            "/query", json={"question": long_question}, content_type="application/json"
        )

        assert response.status_code == 400

    def test_query_not_json(self, client):
        """Test query with non-JSON request."""
        response = client.post("/query", data="not json")
        data = json.loads(response.data)

        assert response.status_code == 400
        assert "error" in data

    def test_index_sample_documents(self, client, clean_store):
        """Test indexing sample FAQ documents."""
        response = client.post(
            "/index", json={"clear_existing": True}, content_type="application/json"
        )
        data = json.loads(response.data)

        assert response.status_code == 200
        assert "chunks_added" in data
        assert data["chunks_added"] > 0
        assert "total_chunks" in data

    def test_documents_list_empty(self, client, clean_store):
        """Test document listing when empty."""
        response = client.get("/documents")
        data = json.loads(response.data)

        assert response.status_code == 200
        assert data["documents"] == []
        assert data["total"] == 0

    def test_documents_list_with_filter(self, client):
        """Test document listing with category filter."""
        response = client.get("/documents?category=leave_policies")
        data = json.loads(response.data)

        assert response.status_code == 200
        assert "documents" in data

    def test_delete_nonexistent_document(self, client):
        """Test deleting document that doesn't exist."""
        response = client.delete("/documents/nonexistent_source")
        data = json.loads(response.data)

        assert response.status_code == 404
        assert "error" in data

    def test_clear_endpoint(self, client):
        """Test clearing the vector store."""
        response = client.delete("/clear")
        data = json.loads(response.data)

        assert response.status_code == 200
        assert "message" in data

    def test_404_endpoint(self, client):
        """Test 404 for unknown endpoint."""
        response = client.get("/unknown_endpoint")
        data = json.loads(response.data)

        assert response.status_code == 404
        assert "error" in data


class TestQueryFlow:
    """Integration tests for complete query flow."""

    @pytest.fixture
    def indexed_client(self, client, clean_store):
        """Client with indexed documents."""
        client.post(
            "/index", json={"clear_existing": True}, content_type="application/json"
        )
        return client

    def test_complete_query_flow(self, indexed_client):
        """Test complete query with indexed documents."""
        response = indexed_client.post(
            "/query",
            json={"question": "How many vacation days do I get?"},
            content_type="application/json",
        )
        data = json.loads(response.data)

        assert response.status_code == 200
        assert "answer" in data
        assert "sources" in data
        assert "confidence" in data
        assert len(data["sources"]) > 0

    def test_query_with_filters(self, indexed_client):
        """Test query with metadata filters."""
        response = indexed_client.post(
            "/query",
            json={
                "question": "policy information",
                "filters": {"category": "leave_policies"},
            },
            content_type="application/json",
        )
        data = json.loads(response.data)

        assert response.status_code == 200
        # Results should match filter
        for source in data.get("sources", []):
            if "category" in source:
                assert source["category"] == "leave_policies"

    def test_query_response_fields(self, indexed_client):
        """Test that query response has all fields."""
        response = indexed_client.post(
            "/query", json={"question": "vacation"}, content_type="application/json"
        )
        data = json.loads(response.data)

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
            assert field in data

    def test_stats_after_queries(self, indexed_client):
        """Test statistics are updated after queries."""
        # Make some queries
        for _ in range(3):
            indexed_client.post(
                "/query", json={"question": "vacation"}, content_type="application/json"
            )

        response = indexed_client.get("/stats")
        data = json.loads(response.data)

        assert data["statistics"]["total_queries"] == 3

    def test_documents_listed_after_index(self, indexed_client):
        """Test documents are listed after indexing."""
        response = indexed_client.get("/documents")
        data = json.loads(response.data)

        assert data["total"] > 0
        assert len(data["documents"]) > 0


class TestUploadEndpoint:
    """Tests for document upload functionality."""

    def test_upload_no_file(self, client):
        """Test upload without file."""
        response = client.post("/upload")
        data = json.loads(response.data)

        assert response.status_code == 400
        assert "error" in data

    def test_upload_empty_filename(self, client):
        """Test upload with empty filename."""
        from io import BytesIO

        response = client.post(
            "/upload", data={"file": (BytesIO(b"content"), "")}
        )
        data = json.loads(response.data)

        assert response.status_code == 400

    def test_upload_unsupported_type(self, client):
        """Test upload with unsupported file type."""
        from io import BytesIO

        response = client.post(
            "/upload", data={"file": (BytesIO(b"content"), "test.exe")}
        )
        data = json.loads(response.data)

        assert response.status_code == 400
        assert "unsupported" in data["error"].lower()

    def test_upload_text_file(self, client, clean_store):
        """Test uploading a text file."""
        from io import BytesIO

        content = b"This is a test HR policy document about leave."
        response = client.post(
            "/upload",
            data={
                "file": (BytesIO(content), "test_policy.txt"),
                "category": "leave_policies",
                "department": "hr",
            },
        )
        data = json.loads(response.data)

        assert response.status_code == 201
        assert "document" in data
        assert "chunks_added" in data
        assert data["chunks_added"] > 0


class TestErrorHandling:
    """Tests for error handling."""

    def test_method_not_allowed(self, client):
        """Test method not allowed error."""
        response = client.put("/query", json={})

        assert response.status_code == 405

    def test_malformed_json(self, client):
        """Test with malformed JSON."""
        response = client.post(
            "/query", data="{malformed", content_type="application/json"
        )

        assert response.status_code in [400, 500]
