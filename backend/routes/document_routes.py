"""Document management API routes."""

import os
import time
from flask import Blueprint, request, jsonify, current_app
from middleware.auth import auth_required, permission_required, get_current_user
from middleware.rate_limiter import rate_limit

documents_bp = Blueprint("documents", __name__, url_prefix="/api/documents")


def get_document_service():
    """Get document service from app context."""
    return current_app.config["DOCUMENT_SERVICE"]


@documents_bp.route("", methods=["GET"])
@auth_required
@permission_required("read")
@rate_limit("100/hour")
def list_documents():
    """
    List all documents with optional filtering.

    Query parameters:
    - category: Filter by category
    - department: Filter by department
    - sort_by: Sort field (created_at, title, chunk_count)
    - limit: Max results (default 100)
    """
    category = request.args.get("category")
    department = request.args.get("department")
    sort_by = request.args.get("sort_by", "created_at")
    limit = min(int(request.args.get("limit", 100)), 500)

    doc_service = get_document_service()
    documents = doc_service.list_documents(
        category=category, department=department, sort_by=sort_by, limit=limit
    )

    return jsonify({"documents": documents, "total": len(documents)}), 200


@documents_bp.route("/<source_id>", methods=["GET"])
@auth_required
@permission_required("read")
def get_document(source_id):
    """Get document details by source ID."""
    doc_service = get_document_service()
    document = doc_service.get_document(source_id)

    if document:
        return jsonify(document), 200
    else:
        return jsonify({"error": "Document not found"}), 404


@documents_bp.route("", methods=["POST"])
@auth_required
@permission_required("write")
@rate_limit("10/hour")
def upload_document():
    """
    Upload and index a new document.

    Request: multipart/form-data
    - file: Document file (PDF, DOCX, TXT, etc.)
    - category: Document category (optional)
    - department: Department (optional)
    - title: Custom title (optional)
    - description: Document description (optional)
    - tags: JSON array of tags (optional)
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    user = get_current_user()

    # Gather metadata from form
    metadata = {
        "category": request.form.get("category", "general"),
        "department": request.form.get("department", "all"),
        "title": request.form.get("title"),
        "description": request.form.get("description"),
    }

    # Parse tags if provided
    tags_str = request.form.get("tags")
    if tags_str:
        try:
            import json

            metadata["tags"] = json.loads(tags_str)
        except json.JSONDecodeError:
            metadata["tags"] = []

    start_time = time.time()
    doc_service = get_document_service()
    success, result = doc_service.upload_and_index(
        file, user_id=user.id if user else None, metadata=metadata
    )

    processing_time = (time.time() - start_time) * 1000

    if success:
        result["processing_time_ms"] = round(processing_time, 2)
        return jsonify({"message": "Document uploaded successfully", **result}), 201
    else:
        return jsonify(result), 400


@documents_bp.route("/<source_id>", methods=["DELETE"])
@auth_required
@permission_required("delete")
@rate_limit("20/hour")
def delete_document(source_id):
    """Delete a document by source ID."""
    user = get_current_user()
    doc_service = get_document_service()
    success, result = doc_service.delete_document(
        source_id, user_id=user.id if user else None
    )

    if success:
        return jsonify(result), 200
    else:
        return jsonify(result), 404


@documents_bp.route("/<source_id>", methods=["PATCH"])
@auth_required
@permission_required("write")
@rate_limit("50/hour")
def update_document_metadata(source_id):
    """
    Update document metadata.

    Request body:
    {
        "title": "New title",
        "category": "new_category",
        "department": "engineering",
        "tags": ["tag1", "tag2"],
        "description": "Updated description"
    }
    """
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    metadata = request.get_json()
    doc_service = get_document_service()
    success, result = doc_service.update_metadata(source_id, metadata)

    if success:
        return jsonify({"message": "Document updated", "document": result}), 200
    else:
        return jsonify(result), 404


@documents_bp.route("/statistics", methods=["GET"])
@auth_required
@permission_required("read")
def get_statistics():
    """Get document statistics."""
    doc_service = get_document_service()
    stats = doc_service.get_statistics()
    return jsonify(stats), 200


@documents_bp.route("/index-sample", methods=["POST"])
@auth_required
@permission_required("write")
@rate_limit("5/hour")
def index_sample():
    """
    Index sample FAQ documents for testing.

    Request body (optional):
    {
        "clear_existing": false
    }
    """
    data = request.get_json() if request.is_json else {}
    clear_existing = data.get("clear_existing", False)

    rag_service = current_app.config["RAG_SERVICE"]

    if clear_existing:
        vector_store = current_app.config["VECTOR_STORE"]
        vector_store.clear()
        rag_service.reset()

    faq_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "sample_faq.txt"
    )

    try:
        start_time = time.time()
        chunks_added = rag_service.index_sample_documents(faq_path)
        processing_time = (time.time() - start_time) * 1000

        return (
            jsonify(
                {
                    "message": "Sample documents indexed successfully",
                    "chunks_added": chunks_added,
                    "processing_time_ms": round(processing_time, 2),
                }
            ),
            200,
        )
    except Exception as e:
        return jsonify({"error": f"Failed to index sample: {str(e)}"}), 500
