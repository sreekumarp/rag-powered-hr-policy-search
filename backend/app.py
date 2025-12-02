"""
HR Assistant RAG - Enterprise Flask Application (Phase 3)

Features:
- JWT and API Key authentication
- Role-based access control (RBAC)
- Rate limiting with token bucket
- SQLAlchemy ORM with PostgreSQL/SQLite support
- Document upload and management
- Enhanced RAG pipeline with query analytics
- Comprehensive logging and monitoring
"""

import os
from flask import Flask, g
from flask_cors import CORS
from config import get_config
from models import db, migrate, init_db
from middleware.rate_limiter import add_rate_limit_headers
from routes import auth_bp, documents_bp, query_bp, admin_bp
from utils.vector_store import VectorStore
from utils.logger import setup_logger
from services import RAGService, DocumentService


def create_app(config_name=None):
    """Application factory pattern."""
    app = Flask(__name__)

    # Load configuration
    if config_name is None:
        config_name = os.environ.get("FLASK_ENV", "development")

    config_class = get_config()
    app.config.from_object(config_class)

    # Setup logging
    logger = setup_logger(
        name="hr_assistant",
        log_dir=app.config["LOG_DIR"],
        log_level=app.config["LOG_LEVEL"],
    )
    app.logger.handlers = logger.handlers
    app.logger.setLevel(logger.level)

    # Initialize database
    init_db(app)

    # Setup CORS
    CORS(app, origins=app.config["CORS_ORIGINS"])

    # Initialize Vector Store
    logger.info("Initializing Vector Store...")
    vector_store = VectorStore(
        model_name=app.config["EMBEDDING_MODEL"],
        index_path=app.config["FAISS_INDEX_PATH"],
        chunk_size=app.config["CHUNK_SIZE"],
        chunk_overlap=app.config["CHUNK_OVERLAP"],
    )
    app.config["VECTOR_STORE"] = vector_store

    # Initialize LLM Service
    logger.info("Initializing LLM Service (GitHub Models)...")

    llm_config = {
        "LLM_ENABLED": app.config.get("LLM_ENABLED", True),
        "GITHUB_TOKEN": app.config.get("GITHUB_TOKEN", ""),
        "GITHUB_MODELS_ENDPOINT": app.config.get("GITHUB_MODELS_ENDPOINT"),
        "GITHUB_MODELS_MODEL": app.config.get("GITHUB_MODELS_MODEL"),
        "GITHUB_MODELS_MAX_TOKENS": app.config.get("GITHUB_MODELS_MAX_TOKENS"),
        "GITHUB_MODELS_TEMPERATURE": app.config.get("GITHUB_MODELS_TEMPERATURE"),
        "LLM_FALLBACK_TO_TEMPLATE": app.config.get("LLM_FALLBACK_TO_TEMPLATE", True),
    }

    from services.llm_service import LLMService
    llm_service = LLMService(llm_config) if app.config.get("LLM_ENABLED", True) else None
    app.config["LLM_SERVICE"] = llm_service

    # Initialize Response Cache (if enabled)
    response_cache = None
    if app.config.get("LLM_CACHE_ENABLED", True) and llm_service:
        logger.info("Initializing Response Cache...")
        from utils.cache import ResponseCache
        response_cache = ResponseCache(
            embedding_model=vector_store.embedding_model,
            similarity_threshold=app.config.get("LLM_CACHE_SIMILARITY", 0.92),
            default_ttl=app.config.get("LLM_CACHE_TTL", 86400),
            max_size=500,
            cache_file="data/llm_cache.json"
        )
        app.config["RESPONSE_CACHE"] = response_cache

    # Initialize RAG Service (with LLM and cache)
    logger.info("Initializing RAG Service...")
    rag_service = RAGService(
        vector_store=vector_store,
        llm_service=llm_service,
        cache=response_cache,
        config={
            "retrieval_k": app.config["RETRIEVAL_K"],
            "confidence_threshold": app.config["CONFIDENCE_THRESHOLD"],
            "min_confidence_for_answer": app.config["MIN_CONFIDENCE_FOR_ANSWER"],
            "enable_llm": app.config.get("LLM_ENABLED", True),
            "enable_citations": app.config.get("LLM_ENABLE_CITATIONS", True),
        },
    )
    app.config["RAG_SERVICE"] = rag_service

    # Initialize Document Service
    logger.info("Initializing Document Service...")
    document_service = DocumentService(
        upload_folder=app.config["UPLOAD_FOLDER"],
        vector_store=vector_store,
        allowed_extensions=app.config["ALLOWED_EXTENSIONS"],
    )
    app.config["DOCUMENT_SERVICE"] = document_service

    # Register blueprints
    app.register_blueprint(auth_bp)
    app.register_blueprint(documents_bp)
    app.register_blueprint(query_bp)
    app.register_blueprint(admin_bp)

    # Request hooks
    @app.before_request
    def before_request():
        """Set request start time for latency tracking."""
        import time

        g.start_time = time.time()

    @app.after_request
    def after_request(response):
        """Add rate limit headers and log request."""
        # Add rate limit headers
        response = add_rate_limit_headers(response)

        # Log request
        if hasattr(g, "start_time"):
            import time

            latency_ms = (time.time() - g.start_time) * 1000
            from flask import request

            app.logger.debug(
                f"{request.method} {request.path} - {response.status_code} - {latency_ms:.2f}ms"
            )

        return response

    # Error handlers
    @app.errorhandler(400)
    def bad_request(error):
        from flask import jsonify

        return jsonify({"error": "Bad request", "message": str(error)}), 400

    @app.errorhandler(401)
    def unauthorized(error):
        from flask import jsonify

        return jsonify({"error": "Unauthorized", "message": str(error)}), 401

    @app.errorhandler(403)
    def forbidden(error):
        from flask import jsonify

        return jsonify({"error": "Forbidden", "message": str(error)}), 403

    @app.errorhandler(404)
    def not_found(error):
        from flask import jsonify

        return jsonify({"error": "Not found", "message": str(error)}), 404

    @app.errorhandler(413)
    def request_entity_too_large(error):
        from flask import jsonify

        max_mb = app.config.get("MAX_CONTENT_LENGTH", 0) / 1024 / 1024
        return (
            jsonify({"error": f"File too large. Maximum size is {max_mb}MB"}),
            413,
        )

    @app.errorhandler(429)
    def ratelimit_handler(error):
        from flask import jsonify

        return jsonify({"error": "Rate limit exceeded", "message": str(error)}), 429

    @app.errorhandler(500)
    def internal_server_error(error):
        from flask import jsonify

        app.logger.error(f"Internal server error: {error}")
        return jsonify({"error": "Internal server error"}), 500

    # Shell context for flask shell
    @app.shell_context_processor
    def make_shell_context():
        from models import User, Document, QueryLog, APIKey

        return {
            "db": db,
            "User": User,
            "Document": Document,
            "QueryLog": QueryLog,
            "APIKey": APIKey,
            "vector_store": vector_store,
            "rag_service": rag_service,
            "document_service": document_service,
        }

    logger.info("Application initialized successfully")
    return app


def create_default_admin(app):
    """Create default admin user if none exists."""
    with app.app_context():
        from models import User

        admin = User.query.filter_by(role="admin").first()
        if not admin:
            admin = User(
                username="admin",
                email="admin@example.com",
                role="admin",
            )
            admin.set_password(os.environ.get("ADMIN_PASSWORD", "admin123"))
            db.session.add(admin)
            db.session.commit()
            app.logger.info("Created default admin user (username: admin)")


if __name__ == "__main__":
    app = create_app()

    # Create default admin user
    create_default_admin(app)

    port = int(os.environ.get("PORT", 51425))
    debug = app.config.get("DEBUG", False)

    print("\n" + "=" * 80)
    print("HR Assistant RAG - Enterprise Edition v3.0.0")
    print("=" * 80)
    print(f"\nEnvironment: {os.environ.get('FLASK_ENV', 'development')}")
    print(f"Debug mode: {debug}")
    print(f"Port: {port}")
    print(f"Database: {app.config['SQLALCHEMY_DATABASE_URI']}")
    print(f"Rate limiting: {app.config['RATE_LIMIT_ENABLED']}")
    print(f"\nFeatures:")
    print("  ✓ JWT & API Key Authentication")
    print("  ✓ Role-Based Access Control")
    print("  ✓ Rate Limiting")
    print("  ✓ Document Upload & Management")
    print("  ✓ Semantic Search with RAG")
    print("  ✓ Query Analytics & Logging")
    print("  ✓ Admin Dashboard APIs")
    print(f"\nAPI Endpoints:")
    print("  Authentication:  /api/auth/*")
    print("  Documents:       /api/documents/*")
    print("  Query:           /api/query")
    print("  Analytics:       /api/analytics")
    print("  Admin:           /api/admin/*")
    print("  Health:          /api/health")
    print("  Info:            /api/info")
    print("\n" + "=" * 80)
    print(f"\nServer starting on http://0.0.0.0:{port}")
    print("Press CTRL+C to quit\n")

    app.run(host="0.0.0.0", port=port, debug=debug)
