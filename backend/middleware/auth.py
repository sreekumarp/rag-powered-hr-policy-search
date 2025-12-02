"""Authentication middleware for JWT and API key validation."""

import jwt
from functools import wraps
from datetime import datetime, timedelta
from flask import request, jsonify, current_app, g


def create_access_token(user_id, role="user", expires_delta=None):
    """Create a JWT access token."""
    if expires_delta is None:
        expires_delta = current_app.config.get("JWT_ACCESS_TOKEN_EXPIRES", timedelta(hours=24))

    payload = {
        "sub": user_id,
        "role": role,
        "iat": datetime.utcnow(),
        "exp": datetime.utcnow() + expires_delta,
        "type": "access",
    }

    return jwt.encode(
        payload,
        current_app.config["JWT_SECRET_KEY"],
        algorithm="HS256",
    )


def create_refresh_token(user_id, expires_delta=None):
    """Create a JWT refresh token."""
    if expires_delta is None:
        expires_delta = current_app.config.get("JWT_REFRESH_TOKEN_EXPIRES", timedelta(days=30))

    payload = {
        "sub": user_id,
        "iat": datetime.utcnow(),
        "exp": datetime.utcnow() + expires_delta,
        "type": "refresh",
    }

    return jwt.encode(
        payload,
        current_app.config["JWT_SECRET_KEY"],
        algorithm="HS256",
    )


def decode_token(token):
    """Decode and validate a JWT token."""
    try:
        payload = jwt.decode(
            token,
            current_app.config["JWT_SECRET_KEY"],
            algorithms=["HS256"],
        )
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


def get_token_from_header():
    """Extract JWT token from Authorization header."""
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        return auth_header[7:]
    return None


def get_api_key_from_header():
    """Extract API key from X-API-Key header."""
    return request.headers.get("X-API-Key", None)


def get_current_user():
    """Get the current authenticated user from request context."""
    return getattr(g, "current_user", None)


def get_current_identity():
    """Get the current identity (user or API key) from request context."""
    return getattr(g, "current_identity", None)


def jwt_required(f):
    """Decorator to require valid JWT token."""

    @wraps(f)
    def decorated(*args, **kwargs):
        token = get_token_from_header()

        if not token:
            return jsonify({"error": "Missing authorization token"}), 401

        payload = decode_token(token)
        if not payload:
            return jsonify({"error": "Invalid or expired token"}), 401

        if payload.get("type") != "access":
            return jsonify({"error": "Invalid token type"}), 401

        # Store user info in request context
        g.current_identity = {
            "type": "jwt",
            "user_id": payload["sub"],
            "role": payload.get("role", "user"),
        }

        # Load user from database
        from models import User

        user = User.query.get(payload["sub"])
        if not user or not user.is_active:
            return jsonify({"error": "User not found or inactive"}), 401

        g.current_user = user

        return f(*args, **kwargs)

    return decorated


def api_key_required(f):
    """Decorator to require valid API key."""

    @wraps(f)
    def decorated(*args, **kwargs):
        api_key = get_api_key_from_header()

        if not api_key:
            return jsonify({"error": "Missing API key"}), 401

        # Find and validate API key
        from models import APIKey

        # Check all active API keys
        all_keys = APIKey.query.filter_by(is_active=True).all()
        valid_key = None

        for key_obj in all_keys:
            if APIKey.verify_key(api_key, key_obj.key_hash):
                valid_key = key_obj
                break

        if not valid_key:
            return jsonify({"error": "Invalid API key"}), 401

        if not valid_key.is_valid():
            return jsonify({"error": "API key expired or inactive"}), 401

        # Record usage
        valid_key.record_usage()

        # Store API key info in request context
        g.current_identity = {
            "type": "api_key",
            "key_id": valid_key.id,
            "user_id": valid_key.user_id,
            "permissions": valid_key.permissions,
        }

        # Load associated user
        from models import User

        user = User.query.get(valid_key.user_id)
        g.current_user = user

        return f(*args, **kwargs)

    return decorated


def auth_required(f):
    """Decorator to require either JWT or API key authentication."""

    @wraps(f)
    def decorated(*args, **kwargs):
        token = get_token_from_header()
        api_key = get_api_key_from_header()

        if token:
            # Try JWT authentication
            payload = decode_token(token)
            if payload and payload.get("type") == "access":
                from models import User

                user = User.query.get(payload["sub"])
                if user and user.is_active:
                    g.current_identity = {
                        "type": "jwt",
                        "user_id": payload["sub"],
                        "role": payload.get("role", "user"),
                    }
                    g.current_user = user
                    return f(*args, **kwargs)

        if api_key:
            # Try API key authentication
            from models import APIKey

            all_keys = APIKey.query.filter_by(is_active=True).all()
            for key_obj in all_keys:
                if APIKey.verify_key(api_key, key_obj.key_hash):
                    if key_obj.is_valid():
                        key_obj.record_usage()
                        from models import User

                        user = User.query.get(key_obj.user_id)
                        g.current_identity = {
                            "type": "api_key",
                            "key_id": key_obj.id,
                            "user_id": key_obj.user_id,
                            "permissions": key_obj.permissions,
                        }
                        g.current_user = user
                        return f(*args, **kwargs)

        return jsonify({"error": "Authentication required"}), 401

    return decorated


def permission_required(permission):
    """Decorator to require specific permission."""

    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            identity = get_current_identity()

            if not identity:
                return jsonify({"error": "Authentication required"}), 401

            has_permission = False

            if identity["type"] == "jwt":
                # Check user role permissions
                user = get_current_user()
                if user:
                    has_permission = user.has_permission(permission)
            elif identity["type"] == "api_key":
                # Check API key permissions
                has_permission = permission in identity.get("permissions", [])

            if not has_permission:
                return (
                    jsonify({"error": f"Permission denied: {permission} required"}),
                    403,
                )

            return f(*args, **kwargs)

        return decorated

    return decorator


def optional_auth(f):
    """Decorator for optional authentication (enhances response but not required)."""

    @wraps(f)
    def decorated(*args, **kwargs):
        token = get_token_from_header()
        api_key = get_api_key_from_header()

        # Try to authenticate but don't fail if not present
        if token:
            payload = decode_token(token)
            if payload and payload.get("type") == "access":
                from models import User

                user = User.query.get(payload["sub"])
                if user and user.is_active:
                    g.current_identity = {
                        "type": "jwt",
                        "user_id": payload["sub"],
                        "role": payload.get("role", "user"),
                    }
                    g.current_user = user

        if api_key and not get_current_user():
            from models import APIKey

            all_keys = APIKey.query.filter_by(is_active=True).all()
            for key_obj in all_keys:
                if APIKey.verify_key(api_key, key_obj.key_hash):
                    if key_obj.is_valid():
                        key_obj.record_usage()
                        from models import User

                        user = User.query.get(key_obj.user_id)
                        g.current_identity = {
                            "type": "api_key",
                            "key_id": key_obj.id,
                            "user_id": key_obj.user_id,
                            "permissions": key_obj.permissions,
                        }
                        g.current_user = user
                        break

        return f(*args, **kwargs)

    return decorated
