"""Rate limiting middleware using token bucket algorithm."""

import time
from functools import wraps
from flask import request, jsonify, current_app, g
from collections import defaultdict
import threading


class TokenBucket:
    """Token bucket rate limiter implementation."""

    def __init__(self, capacity, refill_rate):
        """
        Initialize token bucket.

        Args:
            capacity: Maximum number of tokens (burst capacity)
            refill_rate: Tokens added per second
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = time.time()
        self.lock = threading.Lock()

    def consume(self, tokens=1):
        """
        Attempt to consume tokens from the bucket.

        Returns:
            (success, wait_time) - whether consumption succeeded and time to wait if not
        """
        with self.lock:
            self._refill()

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True, 0
            else:
                # Calculate wait time
                needed = tokens - self.tokens
                wait_time = needed / self.refill_rate
                return False, wait_time

    def _refill(self):
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

    def get_info(self):
        """Get current bucket state."""
        with self.lock:
            self._refill()
            return {
                "remaining_tokens": int(self.tokens),
                "capacity": self.capacity,
                "refill_rate_per_second": self.refill_rate,
            }


class RateLimiter:
    """Rate limiter manager for the application."""

    def __init__(self):
        self.buckets = {}
        self.lock = threading.Lock()

    def get_bucket(self, key, capacity, refill_rate):
        """Get or create a token bucket for the given key."""
        with self.lock:
            if key not in self.buckets:
                self.buckets[key] = TokenBucket(capacity, refill_rate)
            return self.buckets[key]

    def check_rate_limit(self, key, capacity, refill_rate):
        """
        Check if request is within rate limit.

        Returns:
            (allowed, info) - whether request is allowed and rate limit info
        """
        bucket = self.get_bucket(key, capacity, refill_rate)
        success, wait_time = bucket.consume()

        info = bucket.get_info()
        info["wait_time_seconds"] = round(wait_time, 2) if not success else 0

        return success, info

    def clear_bucket(self, key):
        """Remove a bucket (useful for testing or cleanup)."""
        with self.lock:
            if key in self.buckets:
                del self.buckets[key]

    def clear_all(self):
        """Clear all buckets."""
        with self.lock:
            self.buckets.clear()


# Global rate limiter instance
rate_limiter = RateLimiter()


def parse_rate_limit(limit_string):
    """
    Parse rate limit string (e.g., "100/hour", "10/minute", "5/second").

    Returns:
        (capacity, refill_rate) tuple
    """
    parts = limit_string.lower().split("/")
    if len(parts) != 2:
        raise ValueError(f"Invalid rate limit format: {limit_string}")

    count = int(parts[0])
    period = parts[1].strip()

    # Convert period to seconds
    period_seconds = {
        "second": 1,
        "sec": 1,
        "s": 1,
        "minute": 60,
        "min": 60,
        "m": 60,
        "hour": 3600,
        "hr": 3600,
        "h": 3600,
        "day": 86400,
        "d": 86400,
    }

    if period not in period_seconds:
        raise ValueError(f"Unknown period: {period}")

    seconds = period_seconds[period]
    refill_rate = count / seconds

    return count, refill_rate


def get_rate_limit_key():
    """Get the rate limit key for the current request."""
    # Priority: authenticated user > API key > IP address
    identity = getattr(g, "current_identity", None)

    if identity:
        if identity["type"] == "jwt":
            return f"user:{identity['user_id']}"
        elif identity["type"] == "api_key":
            return f"apikey:{identity['key_id']}"

    # Fall back to IP address
    return f"ip:{request.remote_addr}"


def rate_limit(limit_string=None):
    """
    Decorator to apply rate limiting to an endpoint.

    Usage:
        @rate_limit("100/hour")
        def my_endpoint():
            ...

        @rate_limit()  # Uses default from config
        def my_endpoint():
            ...
    """

    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            # Check if rate limiting is enabled
            if not current_app.config.get("RATE_LIMIT_ENABLED", True):
                return f(*args, **kwargs)

            # Get rate limit
            if limit_string:
                limit = limit_string
            else:
                limit = current_app.config.get("RATE_LIMIT_DEFAULT", "100/hour")

            # Parse rate limit
            try:
                capacity, refill_rate = parse_rate_limit(limit)
            except ValueError as e:
                current_app.logger.error(f"Invalid rate limit: {e}")
                return f(*args, **kwargs)

            # Get rate limit key
            key = f"{get_rate_limit_key()}:{request.endpoint}"

            # Check rate limit
            allowed, info = rate_limiter.check_rate_limit(key, capacity, refill_rate)

            # Add rate limit headers
            g.rate_limit_info = info

            if not allowed:
                response = jsonify(
                    {
                        "error": "Rate limit exceeded",
                        "retry_after_seconds": info["wait_time_seconds"],
                        "limit": limit,
                    }
                )
                response.status_code = 429
                response.headers["Retry-After"] = str(int(info["wait_time_seconds"]))
                response.headers["X-RateLimit-Limit"] = str(info["capacity"])
                response.headers["X-RateLimit-Remaining"] = str(info["remaining_tokens"])
                return response

            return f(*args, **kwargs)

        return decorated

    return decorator


def add_rate_limit_headers(response):
    """Add rate limit headers to response (call in after_request)."""
    info = getattr(g, "rate_limit_info", None)
    if info:
        response.headers["X-RateLimit-Limit"] = str(info["capacity"])
        response.headers["X-RateLimit-Remaining"] = str(info["remaining_tokens"])
    return response
