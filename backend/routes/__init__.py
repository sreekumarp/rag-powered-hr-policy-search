"""API routes package."""

from .auth_routes import auth_bp
from .document_routes import documents_bp
from .query_routes import query_bp
from .admin_routes import admin_bp

__all__ = ["auth_bp", "documents_bp", "query_bp", "admin_bp"]
