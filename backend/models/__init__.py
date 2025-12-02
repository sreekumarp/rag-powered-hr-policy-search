"""Database models package."""

from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

db = SQLAlchemy()
migrate = Migrate()


def init_db(app):
    """Initialize database with Flask app."""
    db.init_app(app)
    migrate.init_app(app, db)

    with app.app_context():
        db.create_all()


from .user import User
from .document import Document
from .query_log import QueryLog
from .api_key import APIKey

__all__ = ["db", "migrate", "init_db", "User", "Document", "QueryLog", "APIKey"]
