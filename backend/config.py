"""
Configuration management for HR Assistant RAG API.
Environment-based configuration with sensible defaults.
"""

import os
from datetime import timedelta


class Config:
    """Base configuration."""

    # App
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")
    JSON_SORT_KEYS = False

    # Database
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URL", "sqlite:///data/hr_assistant.db"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {
        "pool_recycle": 300,
        "pool_pre_ping": True,
    }

    # JWT Authentication
    JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "jwt-secret-change-in-production")
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(
        hours=int(os.environ.get("JWT_ACCESS_TOKEN_HOURS", 24))
    )
    JWT_REFRESH_TOKEN_EXPIRES = timedelta(
        days=int(os.environ.get("JWT_REFRESH_TOKEN_DAYS", 30))
    )

    # Rate Limiting
    RATE_LIMIT_ENABLED = os.environ.get("RATE_LIMIT_ENABLED", "true").lower() == "true"
    RATE_LIMIT_DEFAULT = os.environ.get("RATE_LIMIT_DEFAULT", "100/hour")
    RATE_LIMIT_QUERY = os.environ.get("RATE_LIMIT_QUERY", "50/hour")
    RATE_LIMIT_UPLOAD = os.environ.get("RATE_LIMIT_UPLOAD", "10/hour")
    RATE_LIMIT_STORAGE_URL = os.environ.get("RATE_LIMIT_STORAGE_URL", "memory://")

    # File Upload
    MAX_CONTENT_LENGTH = int(os.environ.get("MAX_UPLOAD_SIZE_MB", 50)) * 1024 * 1024
    UPLOAD_FOLDER = os.environ.get("UPLOAD_FOLDER", "uploads")
    ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt", ".html", ".md"}

    # Vector Store / RAG
    EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    FAISS_INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", "data/faiss_index")
    CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 500))
    CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 50))
    RETRIEVAL_K = int(os.environ.get("RETRIEVAL_K", 3))
    CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", 0.3))
    MIN_CONFIDENCE_FOR_ANSWER = float(os.environ.get("MIN_CONFIDENCE", 0.5))

    # Logging
    LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
    LOG_DIR = os.environ.get("LOG_DIR", "logs")

    # CORS
    CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*").split(",")

    # API Keys (comma-separated list)
    API_KEYS = [k.strip() for k in os.environ.get("API_KEYS", "").split(",") if k.strip()]


class DevelopmentConfig(Config):
    """Development configuration."""

    DEBUG = True
    TESTING = False
    LOG_LEVEL = "DEBUG"
    RATE_LIMIT_ENABLED = False


class TestingConfig(Config):
    """Testing configuration."""

    DEBUG = True
    TESTING = True
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"
    RATE_LIMIT_ENABLED = False
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(seconds=5)


class ProductionConfig(Config):
    """Production configuration."""

    DEBUG = False
    TESTING = False
    LOG_LEVEL = "WARNING"

    # Override with secure defaults
    @classmethod
    def init_app(cls, app):
        Config.init_app(app)

        # Ensure required environment variables are set
        required_vars = ["SECRET_KEY", "JWT_SECRET_KEY", "DATABASE_URL"]
        missing = [var for var in required_vars if not os.environ.get(var)]
        if missing:
            raise ValueError(f"Missing required environment variables: {missing}")


config = {
    "development": DevelopmentConfig,
    "testing": TestingConfig,
    "production": ProductionConfig,
    "default": DevelopmentConfig,
}


def get_config():
    """Get configuration based on environment."""
    env = os.environ.get("FLASK_ENV", "development")
    return config.get(env, config["default"])
