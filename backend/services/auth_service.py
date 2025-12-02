"""Authentication service for user management."""

from datetime import datetime
from models import db, User, APIKey
from middleware.auth import create_access_token, create_refresh_token, decode_token


class AuthService:
    """Service for authentication and user management operations."""

    @staticmethod
    def register_user(username, email, password, role="user"):
        """
        Register a new user.

        Returns:
            (success, result) tuple
        """
        # Check if username exists
        if User.query.filter_by(username=username).first():
            return False, {"error": "Username already exists"}

        # Check if email exists
        if User.query.filter_by(email=email).first():
            return False, {"error": "Email already exists"}

        # Validate password strength
        if len(password) < 8:
            return False, {"error": "Password must be at least 8 characters"}

        # Create user
        user = User(username=username, email=email, role=role)
        user.set_password(password)

        db.session.add(user)
        db.session.commit()

        return True, user.to_dict()

    @staticmethod
    def authenticate_user(username_or_email, password):
        """
        Authenticate user and return tokens.

        Returns:
            (success, result) tuple
        """
        # Find user by username or email
        user = User.query.filter(
            (User.username == username_or_email) | (User.email == username_or_email)
        ).first()

        if not user:
            return False, {"error": "Invalid credentials"}

        if not user.check_password(password):
            return False, {"error": "Invalid credentials"}

        if not user.is_active:
            return False, {"error": "Account is inactive"}

        # Update last login
        user.update_last_login()

        # Generate tokens
        access_token = create_access_token(user.id, user.role)
        refresh_token = create_refresh_token(user.id)

        return True, {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "Bearer",
            "user": user.to_dict(),
        }

    @staticmethod
    def refresh_access_token(refresh_token_str):
        """
        Refresh access token using refresh token.

        Returns:
            (success, result) tuple
        """
        payload = decode_token(refresh_token_str)

        if not payload:
            return False, {"error": "Invalid or expired refresh token"}

        if payload.get("type") != "refresh":
            return False, {"error": "Invalid token type"}

        user = User.query.get(payload["sub"])
        if not user or not user.is_active:
            return False, {"error": "User not found or inactive"}

        # Generate new access token
        access_token = create_access_token(user.id, user.role)

        return True, {
            "access_token": access_token,
            "token_type": "Bearer",
        }

    @staticmethod
    def change_password(user_id, current_password, new_password):
        """
        Change user password.

        Returns:
            (success, result) tuple
        """
        user = User.query.get(user_id)
        if not user:
            return False, {"error": "User not found"}

        if not user.check_password(current_password):
            return False, {"error": "Current password is incorrect"}

        if len(new_password) < 8:
            return False, {"error": "New password must be at least 8 characters"}

        user.set_password(new_password)
        db.session.commit()

        return True, {"message": "Password changed successfully"}

    @staticmethod
    def create_api_key(user_id, name, description=None, permissions=None, expires_in_days=None):
        """
        Create a new API key for a user.

        Returns:
            (success, result) tuple with the plain key (only shown once!)
        """
        user = User.query.get(user_id)
        if not user:
            return False, {"error": "User not found"}

        if not permissions:
            permissions = ["read"]  # Default to read-only

        # Generate the key
        plain_key = APIKey.generate_key()
        key_hash = APIKey.hash_key(plain_key)
        key_prefix = plain_key[:10]

        # Set expiration
        expires_at = None
        if expires_in_days:
            from datetime import timedelta

            expires_at = datetime.utcnow() + timedelta(days=expires_in_days)

        # Create API key record
        api_key = APIKey(
            key_hash=key_hash,
            key_prefix=key_prefix,
            name=name,
            description=description,
            permissions=permissions,
            expires_at=expires_at,
            user_id=user_id,
        )

        db.session.add(api_key)
        db.session.commit()

        result = api_key.to_dict()
        result["key"] = plain_key  # Only returned once!

        return True, result

    @staticmethod
    def revoke_api_key(key_id, user_id):
        """
        Revoke an API key.

        Returns:
            (success, result) tuple
        """
        api_key = APIKey.query.filter_by(id=key_id, user_id=user_id).first()
        if not api_key:
            return False, {"error": "API key not found"}

        api_key.is_active = False
        db.session.commit()

        return True, {"message": "API key revoked successfully"}

    @staticmethod
    def list_api_keys(user_id):
        """List all API keys for a user."""
        api_keys = APIKey.query.filter_by(user_id=user_id).all()
        return [key.to_dict() for key in api_keys]

    @staticmethod
    def get_user_by_id(user_id):
        """Get user by ID."""
        user = User.query.get(user_id)
        return user.to_dict() if user else None

    @staticmethod
    def update_user_role(user_id, new_role, admin_user_id):
        """
        Update user role (admin only).

        Returns:
            (success, result) tuple
        """
        admin = User.query.get(admin_user_id)
        if not admin or admin.role != "admin":
            return False, {"error": "Admin privileges required"}

        user = User.query.get(user_id)
        if not user:
            return False, {"error": "User not found"}

        if new_role not in ["user", "admin", "readonly"]:
            return False, {"error": "Invalid role"}

        user.role = new_role
        db.session.commit()

        return True, user.to_dict()

    @staticmethod
    def deactivate_user(user_id, admin_user_id):
        """
        Deactivate a user account (admin only).

        Returns:
            (success, result) tuple
        """
        admin = User.query.get(admin_user_id)
        if not admin or admin.role != "admin":
            return False, {"error": "Admin privileges required"}

        user = User.query.get(user_id)
        if not user:
            return False, {"error": "User not found"}

        if user.id == admin_user_id:
            return False, {"error": "Cannot deactivate yourself"}

        user.is_active = False
        db.session.commit()

        return True, {"message": f"User {user.username} deactivated"}
