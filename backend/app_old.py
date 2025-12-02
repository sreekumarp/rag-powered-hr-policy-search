"""
HR Assistant RAG - Flask Application (Phase 2: Extended Version)

Enhanced features:
- Document upload endpoint (PDF, DOCX, TXT)
- Metadata filtering
- Structured logging
- Better error handling and validation
"""

import os
import time
from flask import Flask, request, jsonify, g
from flask_cors import CORS
from werkzeug.utils import secure_filename
from utils.vector_store import VectorStore
from utils.rag_pipeline import RAGPipeline
from utils.document_processor import DocumentProcessor
from utils.logger import (
    setup_logger,
    log_api_request,
    log_document_indexed,
    log_error,
)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
app.config["JSON_SORT_KEYS"] = False
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50MB max
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["ALLOWED_EXTENSIONS"] = {".pdf", ".docx", ".doc", ".txt", ".html", ".md"}

# Initialize components
logger = setup_logger(name="hr_assistant", log_dir="logs", log_level="DEBUG")

print("Initializing HR Assistant RAG (Phase 2)...")
vector_store = VectorStore(
    model_name="all-MiniLM-L6-v2",
    index_path="data/faiss_index",
    chunk_size=500,
    chunk_overlap=50,
)

rag_pipeline = RAGPipeline(
    vector_store=vector_store,
    retrieval_k=3,
    confidence_threshold=0.3,
    min_confidence_for_answer=0.5,
)

document_processor = DocumentProcessor(upload_dir=app.config["UPLOAD_FOLDER"])


# Middleware for request logging
@app.before_request
def before_request():
    g.start_time = time.time()


@app.after_request
def after_request(response):
    if hasattr(g, "start_time"):
        latency_ms = (time.time() - g.start_time) * 1000
        log_api_request(
            request.method, request.path, response.status_code, latency_ms
        )
    return response


# Error handlers
@app.errorhandler(400)
def bad_request(error):
    log_error("BadRequest", str(error))
    return jsonify({"error": "Bad request", "message": str(error)}), 400


@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(413)
def file_too_large(error):
    log_error("FileTooLarge", "Upload exceeds 50MB limit")
    return jsonify({"error": "File too large. Maximum size is 50MB"}), 413


@app.errorhandler(500)
def internal_error(error):
    log_error("InternalError", str(error))
    return jsonify({"error": "Internal server error"}), 500


# Routes
@app.route("/", methods=["GET"])
def home():
    """API information."""
    return jsonify(
        {
            "name": "HR Assistant RAG API",
            "version": "2.0.0",
            "phase": "Extended MVP",
            "endpoints": {
                "POST /upload": "Upload documents (PDF, DOCX, TXT)",
                "POST /query": "Query HR policies with optional filters",
                "GET /documents": "List indexed documents",
                "DELETE /documents/<source>": "Remove document by source",
                "POST /index": "Index sample FAQ documents",
                "GET /stats": "System statistics",
                "GET /health": "Health check",
            },
        }
    )


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    stats = vector_store.get_stats()
    return jsonify(
        {
            "status": "healthy",
            "service": "hr-assistant-rag",
            "version": "2.0.0",
            "vector_store_ready": vector_store.index is not None,
            "total_chunks": stats["total_chunks"],
            "total_documents": stats["total_documents"],
        }
    )


@app.route("/upload", methods=["POST"])
def upload_document():
    """
    Upload and index a document.

    Supports: PDF, DOCX, TXT, HTML, MD

    Request: multipart/form-data with 'file' field
    Optional form fields:
    - category: Document category (e.g., "leave_policies")
    - department: Department (e.g., "engineering")
    - title: Custom title (defaults to filename)

    Response:
    {
        "message": "Document uploaded and indexed successfully",
        "document": {...},
        "chunks_added": 10
    }
    """
    # Check if file is present
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # Validate file extension
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in app.config["ALLOWED_EXTENSIONS"]:
        return (
            jsonify(
                {
                    "error": f"Unsupported file type: {ext}",
                    "supported": list(app.config["ALLOWED_EXTENSIONS"]),
                }
            ),
            400,
        )

    try:
        start_time = time.time()

        # Save uploaded file
        file_path = document_processor.save_uploaded_file(file, file.filename)

        # Process document
        doc_info = document_processor.process_file(file_path)

        # Get metadata from form
        metadata = {
            "category": request.form.get("category", "general"),
            "department": request.form.get("department", "all"),
            "title": request.form.get("title", doc_info["filename"]),
            "file_type": ext,
            "file_size": doc_info["file_size"],
            "file_hash": doc_info["file_hash"],
            "word_count": doc_info["word_count"],
        }

        # Create source identifier
        source = f"{metadata['title']}_{doc_info['file_hash'][:8]}"

        # Index document
        result = vector_store.add_document(
            doc_info["extracted_text"], source, metadata
        )

        processing_time = (time.time() - start_time) * 1000
        log_document_indexed(file.filename, result["chunks_added"], processing_time)

        return (
            jsonify(
                {
                    "message": "Document uploaded and indexed successfully",
                    "document": {
                        "source": source,
                        "filename": doc_info["filename"],
                        "file_size_bytes": doc_info["file_size"],
                        "character_count": doc_info["char_count"],
                        "word_count": doc_info["word_count"],
                    },
                    "chunks_added": result["chunks_added"],
                    "total_chunks": result["total_chunks"],
                    "processing_time_ms": round(processing_time, 2),
                }
            ),
            201,
        )

    except Exception as e:
        log_error("UploadError", str(e), filename=file.filename)
        return jsonify({"error": f"Failed to process document: {str(e)}"}), 500


@app.route("/query", methods=["POST"])
def query():
    """
    Query HR policies with optional filtering.

    Request body:
    {
        "question": "How many vacation days do I get?",
        "filters": {
            "category": "leave_policies",
            "department": "engineering"
        },
        "include_low_confidence": false
    }

    Response:
    {
        "answer": "...",
        "sources": [...],
        "confidence": 0.85,
        "relevant_chunks": 3,
        "query": "...",
        "filters_applied": {...},
        "processing_time_ms": 150,
        "warnings": []
    }
    """
    # Validate request
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()

    if "question" not in data:
        return jsonify({"error": "Missing 'question' field"}), 400

    question = data["question"].strip()

    # Validate question
    if not question:
        return jsonify({"error": "Question cannot be empty"}), 400

    if len(question) < 3:
        return jsonify({"error": "Question too short (minimum 3 characters)"}), 400

    if len(question) > 1000:
        return jsonify({"error": "Question too long (maximum 1000 characters)"}), 400

    # Check vector store
    if vector_store.index is None or len(vector_store.documents) == 0:
        return (
            jsonify(
                {
                    "error": "No documents indexed",
                    "hint": "Upload documents using POST /upload or index sample using POST /index",
                }
            ),
            400,
        )

    # Get optional parameters
    filters = data.get("filters", None)
    include_low_confidence = data.get("include_low_confidence", False)

    # Process query
    try:
        result = rag_pipeline.query(question, filters, include_low_confidence)
        return jsonify(result), 200
    except Exception as e:
        log_error("QueryError", str(e), question=question[:50])
        return jsonify({"error": f"Failed to process query: {str(e)}"}), 500


@app.route("/documents", methods=["GET"])
def list_documents():
    """
    List all indexed documents with metadata.

    Query parameters:
    - category: Filter by category
    - sort_by: Sort field (indexed_at, chunk_count)

    Response:
    {
        "documents": [...],
        "total": 10
    }
    """
    documents = vector_store.get_document_list()

    # Apply filters
    category_filter = request.args.get("category")
    if category_filter:
        documents = [d for d in documents if d.get("category") == category_filter]

    # Apply sorting
    sort_by = request.args.get("sort_by", "indexed_at")
    if sort_by in ["indexed_at", "chunk_count"]:
        documents = sorted(documents, key=lambda x: x.get(sort_by, ""), reverse=True)

    return jsonify({"documents": documents, "total": len(documents)}), 200


@app.route("/documents/<source>", methods=["DELETE"])
def delete_document(source):
    """
    Delete a document by source identifier.

    Response:
    {
        "message": "Document deleted successfully",
        "source": "..."
    }
    """
    try:
        success = vector_store.delete_document(source)

        if success:
            return (
                jsonify(
                    {"message": "Document deleted successfully", "source": source}
                ),
                200,
            )
        else:
            return jsonify({"error": f"Document not found: {source}"}), 404
    except Exception as e:
        log_error("DeleteError", str(e), source=source)
        return jsonify({"error": f"Failed to delete document: {str(e)}"}), 500


@app.route("/index", methods=["POST"])
def index_sample():
    """Index sample FAQ documents."""
    data = request.get_json() if request.is_json else {}
    clear_existing = data.get("clear_existing", False)

    if clear_existing:
        vector_store.clear()
        rag_pipeline.reset_statistics()

    faq_path = os.path.join(os.path.dirname(__file__), "data", "sample_faq.txt")

    try:
        start_time = time.time()
        chunks_added = rag_pipeline.add_sample_documents(faq_path)
        processing_time = (time.time() - start_time) * 1000

        if chunks_added > 0:
            log_document_indexed("sample_faq.txt", chunks_added, processing_time)
            return (
                jsonify(
                    {
                        "message": "Sample documents indexed successfully",
                        "chunks_added": chunks_added,
                        "total_chunks": len(vector_store.documents),
                        "processing_time_ms": round(processing_time, 2),
                    }
                ),
                200,
            )
        else:
            return jsonify({"error": "No documents were indexed"}), 500
    except Exception as e:
        log_error("IndexError", str(e))
        return jsonify({"error": f"Failed to index documents: {str(e)}"}), 500


@app.route("/stats", methods=["GET"])
def get_stats():
    """Get comprehensive system statistics."""
    try:
        pipeline_info = rag_pipeline.get_pipeline_info()
        return jsonify(pipeline_info), 200
    except Exception as e:
        log_error("StatsError", str(e))
        return jsonify({"error": f"Failed to get stats: {str(e)}"}), 500


@app.route("/clear", methods=["DELETE"])
def clear_index():
    """Clear all indexed documents and reset statistics."""
    try:
        vector_store.clear()
        rag_pipeline.reset_statistics()
        return jsonify({"message": "Vector store and statistics cleared"}), 200
    except Exception as e:
        log_error("ClearError", str(e))
        return jsonify({"error": f"Failed to clear: {str(e)}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 51425))
    debug = os.environ.get("FLASK_DEBUG", "true").lower() == "true"

    print(f"\nStarting HR Assistant RAG (Phase 2) on port {port}")
    print(f"Debug mode: {debug}")
    print("\nPhase 2 Features:")
    print("  - Document upload (PDF, DOCX, TXT)")
    print("  - Metadata filtering")
    print("  - Structured logging")
    print("  - Query analytics")
    print("\nEndpoints:")
    print("  GET  /            - API information")
    print("  GET  /health      - Health check")
    print("  POST /upload      - Upload document")
    print("  POST /query       - Query with filters")
    print("  GET  /documents   - List documents")
    print("  DELETE /documents/<source> - Remove document")
    print("  POST /index       - Index sample FAQ")
    print("  GET  /stats       - System statistics")
    print("  DELETE /clear     - Clear everything")
    print("\n")

    app.run(host="0.0.0.0", port=port, debug=debug)
