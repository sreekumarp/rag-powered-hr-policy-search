"""API Key model for programmatic access."""

import secrets
from datetime import datetime
from . import db


class APIKey(db.Model):
    """Model for managing API keys for programmatic access."""

    __tablename__ = "api_keys"

    id = db.Column(db.Integer, primary_key=True)
    key_hash = db.Column(db.String(256), unique=True, nullable=False, index=True)
    key_prefix = db.Column(db.String(10), nullable=False)  # First few chars for identification
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text, nullable=True)

    # Permissions
    permissions = db.Column(db.JSON, default=list)  # ["read", "write", "delete"]
    rate_limit = db.Column(db.Integer, default=100)  # requests per hour

    # Usage tracking
    last_used_at = db.Column(db.DateTime, nullable=True)
    usage_count = db.Column(db.Integer, default=0)

    # Status
    is_active = db.Column(db.Boolean, default=True)
    expires_at = db.Column(db.DateTime, nullable=True)

    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Foreign keys
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)

    @staticmethod
    def generate_key():
        """Generate a secure API key."""
        return f"hra_{secrets.token_urlsafe(32)}"

    @staticmethod
    def hash_key(key):
        """Hash an API key for storage."""
        from werkzeug.security import generate_password_hash

        return generate_password_hash(key)

    @staticmethod
    def verify_key(key, key_hash):
        """Verify an API key against its hash."""
        from werkzeug.security import check_password_hash

        return check_password_hash(key_hash, key)

    def record_usage(self):
        """Record that this key was used."""
        self.last_used_at = datetime.utcnow()
        self.usage_count += 1
        db.session.commit()

    def is_valid(self):
        """Check if the key is valid (active and not expired)."""
        if not self.is_active:
            return False
        if self.expires_at and self.expires_at < datetime.utcnow():
            return False
        return True

    def has_permission(self, permission):
        """Check if the key has a specific permission."""
        return permission in self.permissions

    def to_dict(self, include_sensitive=False):
        """Convert to dictionary for JSON serialization."""
        data = {
            "id": self.id,
            "key_prefix": self.key_prefix,
            "name": self.name,
            "description": self.description,
            "permissions": self.permissions,
            "rate_limit": self.rate_limit,
            "is_active": self.is_active,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "last_used_at": self.last_used_at.isoformat() if self.last_used_at else None,
            "usage_count": self.usage_count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "user_id": self.user_id,
        }
        return data

    def __repr__(self):
        return f"<APIKey {self.key_prefix}...>"
