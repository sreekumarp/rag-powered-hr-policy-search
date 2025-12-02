"""Admin API routes for system management."""

from flask import Blueprint, request, jsonify, current_app
from middleware.auth import jwt_required, permission_required, get_current_user
from middleware.rate_limiter import rate_limit
from services.auth_service import AuthService
from models import db, User, Document, QueryLog

admin_bp = Blueprint("admin", __name__, url_prefix="/api/admin")


@admin_bp.route("/users", methods=["GET"])
@jwt_required
@permission_required("admin")
def list_users():
    """List all users (admin only)."""
    users = User.query.all()
    return jsonify({"users": [user.to_dict() for user in users]}), 200


@admin_bp.route("/users/<int:user_id>", methods=["GET"])
@jwt_required
@permission_required("admin")
def get_user(user_id):
    """Get user details (admin only)."""
    user = User.query.get(user_id)
    if not user:
        return jsonify({"error": "User not found"}), 404
    return jsonify(user.to_dict()), 200


@admin_bp.route("/users/<int:user_id>/role", methods=["PATCH"])
@jwt_required
@permission_required("admin")
@rate_limit("10/hour")
def update_user_role(user_id):
    """
    Update user role (admin only).

    Request body:
    {
        "role": "admin"  // or "user" or "readonly"
    }
    """
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    if "role" not in data:
        return jsonify({"error": "Missing role field"}), 400

    admin_user = get_current_user()
    success, result = AuthService.update_user_role(user_id, data["role"], admin_user.id)

    if success:
        return jsonify({"message": "Role updated", "user": result}), 200
    else:
        return jsonify(result), 400


@admin_bp.route("/users/<int:user_id>/deactivate", methods=["POST"])
@jwt_required
@permission_required("admin")
@rate_limit("10/hour")
def deactivate_user(user_id):
    """Deactivate a user account (admin only)."""
    admin_user = get_current_user()
    success, result = AuthService.deactivate_user(user_id, admin_user.id)

    if success:
        return jsonify(result), 200
    else:
        return jsonify(result), 400


@admin_bp.route("/stats", methods=["GET"])
@jwt_required
@permission_required("admin")
def system_stats():
    """Get comprehensive system statistics (admin only)."""
    from sqlalchemy import func

    # User stats
    user_stats = db.session.query(
        func.count(User.id).label("total"),
        func.sum(func.cast(User.is_active, db.Integer)).label("active"),
    ).first()

    # Document stats
    doc_stats = db.session.query(
        func.count(Document.id).label("total"),
        func.sum(Document.chunk_count).label("chunks"),
        func.sum(Document.file_size).label("size"),
    ).first()

    # Query stats
    query_stats = QueryLog.get_statistics(days=30)

    # Vector store stats
    vector_store = current_app.config["VECTOR_STORE"]
    vs_stats = vector_store.get_stats()

    return (
        jsonify(
            {
                "users": {
                    "total": user_stats.total or 0,
                    "active": int(user_stats.active or 0),
                },
                "documents": {
                    "total": doc_stats.total or 0,
                    "total_chunks": int(doc_stats.chunks or 0),
                    "total_size_bytes": int(doc_stats.size or 0),
                },
                "queries": query_stats,
                "vector_store": vs_stats,
            }
        ),
        200,
    )


@admin_bp.route("/clear-data", methods=["DELETE"])
@jwt_required
@permission_required("admin")
@rate_limit("1/hour")
def clear_all_data():
    """
    Clear all data (admin only, dangerous operation).

    Request body:
    {
        "confirm": "DELETE_ALL_DATA"
    }
    """
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    if data.get("confirm") != "DELETE_ALL_DATA":
        return (
            jsonify(
                {
                    "error": "Must confirm deletion with 'confirm': 'DELETE_ALL_DATA'"
                }
            ),
            400,
        )

    try:
        # Clear vector store
        vector_store = current_app.config["VECTOR_STORE"]
        vector_store.clear()

        # Reset RAG service
        rag_service = current_app.config["RAG_SERVICE"]
        rag_service.reset()

        # Clear database tables (except users and API keys)
        QueryLog.query.delete()
        Document.query.delete()
        db.session.commit()

        return jsonify({"message": "All data cleared successfully"}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"Failed to clear data: {str(e)}"}), 500


@admin_bp.route("/logs/queries", methods=["GET"])
@jwt_required
@permission_required("admin")
def query_logs():
    """
    Get all query logs (admin only).

    Query parameters:
    - limit: Max results (default 100)
    - offset: Pagination offset (default 0)
    - status: Filter by status (success, error, low_confidence)
    """
    limit = min(int(request.args.get("limit", 100)), 500)
    offset = int(request.args.get("offset", 0))
    status = request.args.get("status")

    query = QueryLog.query

    if status:
        query = query.filter_by(status=status)

    logs = query.order_by(QueryLog.created_at.desc()).offset(offset).limit(limit).all()

    return jsonify({"logs": [log.to_dict() for log in logs], "count": len(logs)}), 200


@admin_bp.route("/config", methods=["GET"])
@jwt_required
@permission_required("admin")
def get_config():
    """Get current configuration (admin only)."""
    return (
        jsonify(
            {
                "rate_limiting_enabled": current_app.config.get(
                    "RATE_LIMIT_ENABLED", True
                ),
                "max_upload_size_mb": current_app.config.get("MAX_CONTENT_LENGTH", 0)
                / 1024
                / 1024,
                "allowed_extensions": list(
                    current_app.config.get("ALLOWED_EXTENSIONS", [])
                ),
                "embedding_model": current_app.config.get("EMBEDDING_MODEL"),
                "chunk_size": current_app.config.get("CHUNK_SIZE"),
                "chunk_overlap": current_app.config.get("CHUNK_OVERLAP"),
                "retrieval_k": current_app.config.get("RETRIEVAL_K"),
                "confidence_threshold": current_app.config.get("CONFIDENCE_THRESHOLD"),
            }
        ),
        200,
    )
