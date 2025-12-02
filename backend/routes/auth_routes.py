"""Authentication API routes."""

from flask import Blueprint, request, jsonify
from services.auth_service import AuthService
from middleware.auth import jwt_required, get_current_user
from middleware.rate_limiter import rate_limit

auth_bp = Blueprint("auth", __name__, url_prefix="/api/auth")


@auth_bp.route("/register", methods=["POST"])
@rate_limit("10/hour")
def register():
    """
    Register a new user.

    Request body:
    {
        "username": "johndoe",
        "email": "john@example.com",
        "password": "securepassword"
    }
    """
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()

    required_fields = ["username", "email", "password"]
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Missing required field: {field}"}), 400

    username = data["username"].strip()
    email = data["email"].strip().lower()
    password = data["password"]

    # Basic validation
    if len(username) < 3:
        return jsonify({"error": "Username must be at least 3 characters"}), 400
    if "@" not in email:
        return jsonify({"error": "Invalid email format"}), 400

    success, result = AuthService.register_user(username, email, password)

    if success:
        return jsonify({"message": "User registered successfully", "user": result}), 201
    else:
        return jsonify(result), 400


@auth_bp.route("/login", methods=["POST"])
@rate_limit("20/hour")
def login():
    """
    Authenticate user and get tokens.

    Request body:
    {
        "username": "johndoe",  // or email
        "password": "securepassword"
    }
    """
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()

    if "username" not in data or "password" not in data:
        return jsonify({"error": "Missing username or password"}), 400

    success, result = AuthService.authenticate_user(data["username"], data["password"])

    if success:
        return jsonify(result), 200
    else:
        return jsonify(result), 401


@auth_bp.route("/refresh", methods=["POST"])
@rate_limit("50/hour")
def refresh():
    """
    Refresh access token.

    Request body:
    {
        "refresh_token": "..."
    }
    """
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()

    if "refresh_token" not in data:
        return jsonify({"error": "Missing refresh_token"}), 400

    success, result = AuthService.refresh_access_token(data["refresh_token"])

    if success:
        return jsonify(result), 200
    else:
        return jsonify(result), 401


@auth_bp.route("/me", methods=["GET"])
@jwt_required
def get_profile():
    """Get current user profile."""
    user = get_current_user()
    return jsonify(user.to_dict()), 200


@auth_bp.route("/change-password", methods=["POST"])
@jwt_required
@rate_limit("5/hour")
def change_password():
    """
    Change user password.

    Request body:
    {
        "current_password": "...",
        "new_password": "..."
    }
    """
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()

    if "current_password" not in data or "new_password" not in data:
        return jsonify({"error": "Missing current_password or new_password"}), 400

    user = get_current_user()
    success, result = AuthService.change_password(
        user.id, data["current_password"], data["new_password"]
    )

    if success:
        return jsonify(result), 200
    else:
        return jsonify(result), 400


@auth_bp.route("/api-keys", methods=["GET"])
@jwt_required
def list_api_keys():
    """List user's API keys."""
    user = get_current_user()
    keys = AuthService.list_api_keys(user.id)
    return jsonify({"api_keys": keys}), 200


@auth_bp.route("/api-keys", methods=["POST"])
@jwt_required
@rate_limit("5/hour")
def create_api_key():
    """
    Create a new API key.

    Request body:
    {
        "name": "My API Key",
        "description": "For automation scripts",
        "permissions": ["read", "write"],
        "expires_in_days": 90
    }
    """
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()

    if "name" not in data:
        return jsonify({"error": "Missing name field"}), 400

    user = get_current_user()
    success, result = AuthService.create_api_key(
        user_id=user.id,
        name=data["name"],
        description=data.get("description"),
        permissions=data.get("permissions"),
        expires_in_days=data.get("expires_in_days"),
    )

    if success:
        return (
            jsonify(
                {
                    "message": "API key created successfully",
                    "api_key": result,
                    "warning": "Save this key securely. It will not be shown again!",
                }
            ),
            201,
        )
    else:
        return jsonify(result), 400


@auth_bp.route("/api-keys/<int:key_id>", methods=["DELETE"])
@jwt_required
def revoke_api_key(key_id):
    """Revoke an API key."""
    user = get_current_user()
    success, result = AuthService.revoke_api_key(key_id, user.id)

    if success:
        return jsonify(result), 200
    else:
        return jsonify(result), 404
