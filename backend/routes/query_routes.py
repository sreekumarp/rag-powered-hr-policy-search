"""Query API routes for RAG operations."""

from flask import Blueprint, request, jsonify, current_app
from middleware.auth import auth_required, optional_auth, get_current_user
from middleware.rate_limiter import rate_limit

query_bp = Blueprint("query", __name__, url_prefix="/api")


def get_rag_service():
    """Get RAG service from app context."""
    return current_app.config["RAG_SERVICE"]


@query_bp.route("/query", methods=["POST"])
@optional_auth
@rate_limit("50/hour")
def query():
    """
    Query HR policies with semantic search.

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
        "query_id": "uuid",
        "answer": "...",
        "sources": [...],
        "confidence": 0.85,
        "relevant_chunks": 3,
        "processing_time_ms": 150,
        "warnings": []
    }
    """
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

    # Check vector store has data
    vector_store = current_app.config["VECTOR_STORE"]
    if vector_store.index is None or len(vector_store.documents) == 0:
        return (
            jsonify(
                {
                    "error": "No documents indexed",
                    "hint": "Upload documents using POST /api/documents or index sample using POST /api/documents/index-sample",
                }
            ),
            400,
        )

    # Get optional parameters
    filters = data.get("filters")
    include_low_confidence = data.get("include_low_confidence", False)

    # Get user ID if authenticated
    user = get_current_user()
    user_id = user.id if user else None

    # Execute query
    try:
        rag_service = get_rag_service()
        result = rag_service.query(
            question=question,
            filters=filters,
            include_low_confidence=include_low_confidence,
            user_id=user_id,
        )
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": f"Query failed: {str(e)}"}), 500


@query_bp.route("/query/history", methods=["GET"])
@auth_required
def query_history():
    """
    Get query history for current user.

    Query parameters:
    - limit: Max results (default 50)
    - offset: Pagination offset (default 0)
    """
    user = get_current_user()
    limit = min(int(request.args.get("limit", 50)), 200)
    offset = int(request.args.get("offset", 0))

    rag_service = get_rag_service()
    history = rag_service.get_query_history(user_id=user.id, limit=limit, offset=offset)

    return jsonify({"queries": history, "count": len(history)}), 200


@query_bp.route("/analytics", methods=["GET"])
@auth_required
def get_analytics():
    """
    Get query analytics.

    Query parameters:
    - days: Analysis period (default 30)
    """
    days = int(request.args.get("days", 30))
    rag_service = get_rag_service()
    analytics = rag_service.get_analytics(days=days)

    return jsonify(analytics), 200


@query_bp.route("/analytics/popular", methods=["GET"])
@auth_required
def popular_queries():
    """Get most popular query patterns."""
    limit = min(int(request.args.get("limit", 10)), 50)
    rag_service = get_rag_service()
    popular = rag_service.get_popular_queries(limit=limit)

    return jsonify({"popular_queries": popular}), 200


@query_bp.route("/analytics/low-confidence", methods=["GET"])
@auth_required
def low_confidence_queries():
    """Get queries with low confidence scores."""
    threshold = float(request.args.get("threshold", 0.5))
    limit = min(int(request.args.get("limit", 20)), 100)

    rag_service = get_rag_service()
    queries = rag_service.get_low_confidence_queries(threshold=threshold, limit=limit)

    return jsonify({"low_confidence_queries": queries, "threshold": threshold}), 200


@query_bp.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint (no auth required)."""
    vector_store = current_app.config["VECTOR_STORE"]
    stats = vector_store.get_stats()

    return (
        jsonify(
            {
                "status": "healthy",
                "service": "hr-assistant-rag",
                "version": "3.0.0",
                "vector_store_ready": vector_store.index is not None,
                "total_chunks": stats["total_chunks"],
                "total_documents": stats["total_documents"],
            }
        ),
        200,
    )


@query_bp.route("/info", methods=["GET"])
def api_info():
    """API information (no auth required)."""
    return (
        jsonify(
            {
                "name": "HR Assistant RAG API",
                "version": "3.0.0",
                "description": "Enterprise HR policy assistant with RAG capabilities",
                "features": [
                    "JWT and API Key authentication",
                    "Role-based access control",
                    "Rate limiting",
                    "Document upload and indexing",
                    "Semantic search with confidence scoring",
                    "Query analytics and history",
                    "Metadata filtering",
                ],
                "endpoints": {
                    "auth": "/api/auth/*",
                    "documents": "/api/documents/*",
                    "query": "/api/query",
                    "analytics": "/api/analytics",
                    "admin": "/api/admin/*",
                },
            }
        ),
        200,
    )
