"""
LLM Service using GitHub Models (GPT-4o-mini).

Features:
- GitHub Models integration with free tier support
- Automatic fallback to template responses
- Rate limit handling
- Error recovery
"""

import os
import time
import logging
from typing import Dict, Tuple, Optional
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Base class for LLM providers."""

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate response from prompt."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is configured and available."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get provider name."""
        pass


class GitHubModelsProvider(LLMProvider):
    """GitHub Models provider using GPT-4o-mini (free tier)."""

    def __init__(self, token: str, model: str, endpoint: str):
        """
        Initialize GitHub Models provider.

        Args:
            token: GitHub Personal Access Token
            model: Model name (e.g., 'gpt-4o-mini')
            endpoint: API endpoint URL
        """
        self.token = token
        self.model = model
        self.endpoint = endpoint
        self.client = None

        # Rate limit tracking
        self.request_count = 0
        self.last_reset = time.time()
        self.rate_limit_per_day = 50  # GPT-4o-mini free tier
        self.rate_limit_per_minute = 10
        self.minute_requests = []

    def get_name(self) -> str:
        return "github-models"

    def is_available(self) -> bool:
        """Check if GitHub token is configured."""
        return bool(self.token and self.token != "" and not self.token.startswith("ghp_your"))

    def _check_rate_limit(self) -> bool:
        """
        Check if we're within rate limits.

        Returns:
            True if request can proceed, False if rate limited
        """
        now = time.time()

        # Reset daily counter if needed
        if now - self.last_reset > 86400:  # 24 hours
            self.request_count = 0
            self.last_reset = now

        # Check daily limit
        if self.request_count >= self.rate_limit_per_day:
            logger.warning(f"Daily rate limit reached ({self.rate_limit_per_day}/day)")
            return False

        # Check per-minute limit
        self.minute_requests = [t for t in self.minute_requests if now - t < 60]
        if len(self.minute_requests) >= self.rate_limit_per_minute:
            logger.warning(f"Per-minute rate limit reached ({self.rate_limit_per_minute}/min)")
            return False

        return True

    def generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.1) -> str:
        """
        Generate response using GitHub Models.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-1.0)

        Returns:
            Generated text

        Raises:
            Exception: If generation fails
        """
        # Check rate limits
        if not self._check_rate_limit():
            raise Exception("GitHub Models rate limit exceeded")

        # Initialize client lazily
        if not self.client:
            try:
                from openai import OpenAI
                self.client = OpenAI(
                    base_url=self.endpoint,
                    api_key=self.token,
                )
            except ImportError:
                raise Exception("OpenAI SDK not installed. Run: pip install openai")

        try:
            # Make API call
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )

            # Track rate limit
            self.request_count += 1
            self.minute_requests.append(time.time())

            # Extract response
            answer = response.choices[0].message.content

            logger.info(f"GitHub Models generation successful (daily: {self.request_count}/{self.rate_limit_per_day})")

            return answer

        except Exception as e:
            # Handle specific errors
            error_msg = str(e).lower()

            if "rate_limit" in error_msg or "quota" in error_msg:
                logger.error("GitHub Models rate limit exceeded")
                raise Exception("Rate limit exceeded")
            elif "authentication" in error_msg or "unauthorized" in error_msg:
                logger.error("GitHub token authentication failed")
                raise Exception("Authentication failed - check GITHUB_TOKEN")
            else:
                logger.error(f"GitHub Models generation failed: {e}")
                raise


class TemplateFallback(LLMProvider):
    """Fallback provider using template-based responses (no LLM)."""

    def get_name(self) -> str:
        return "template"

    def is_available(self) -> bool:
        return True

    def generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.1) -> str:
        """
        Extract answer from prompt context (simple fallback).

        Looks for context sections in prompt and returns first one.
        """
        lines = prompt.split("\n")

        # Find context section
        in_context = False
        context_lines = []

        for line in lines:
            if "[Source 1]" in line:
                in_context = True
                continue

            if in_context:
                if line.startswith("[Source") or line.startswith("---"):
                    break
                if line.strip():
                    context_lines.append(line.strip())

        if context_lines:
            # Return first context chunk
            return f"Based on HR policies:\n\n{' '.join(context_lines[:3])}"

        return "I don't have enough information to answer this question."


class LLMService:
    """
    LLM service with GitHub Models and template fallback.

    Features:
    - GitHub Models (GPT-4o-mini) as primary provider
    - Template fallback if rate limited or disabled
    - Usage tracking
    - Error handling
    """

    def __init__(self, config: Dict):
        """
        Initialize LLM service.

        Args:
            config: Configuration dict with GitHub token, model, etc.
        """
        self.config = config
        self.enabled = config.get("LLM_ENABLED", True)
        self.fallback_to_template = config.get("LLM_FALLBACK_TO_TEMPLATE", True)

        # Initialize providers
        self.github_provider = GitHubModelsProvider(
            token=config.get("GITHUB_TOKEN", ""),
            model=config.get("GITHUB_MODELS_MODEL", "gpt-4o-mini"),
            endpoint=config.get("GITHUB_MODELS_ENDPOINT", "https://models.inference.ai.azure.com")
        )

        self.template_provider = TemplateFallback()

        # Statistics
        self.total_requests = 0
        self.github_requests = 0
        self.template_requests = 0
        self.cache_hits = 0
        self.errors = 0

    def generate(
        self,
        prompt: str,
        max_tokens: int = None,
        temperature: float = None,
    ) -> Tuple[str, Dict]:
        """
        Generate response with automatic fallback.

        Args:
            prompt: Input prompt
            max_tokens: Max tokens to generate (default from config)
            temperature: Sampling temperature (default from config)

        Returns:
            Tuple of (response_text, metadata)
        """
        max_tokens = max_tokens or self.config.get("GITHUB_MODELS_MAX_TOKENS", 500)
        temperature = temperature or self.config.get("GITHUB_MODELS_TEMPERATURE", 0.1)

        self.total_requests += 1
        start_time = time.time()

        # Try GitHub Models if enabled
        if self.enabled and self.github_provider.is_available():
            try:
                response = self.github_provider.generate(prompt, max_tokens, temperature)
                latency_ms = (time.time() - start_time) * 1000

                self.github_requests += 1

                metadata = {
                    "provider": "github-models",
                    "model": self.config.get("GITHUB_MODELS_MODEL"),
                    "latency_ms": round(latency_ms, 2),
                    "tokens_estimated": len(response.split()) * 1.3,
                    "cached": False,
                    "success": True
                }

                return response, metadata

            except Exception as e:
                logger.error(f"GitHub Models failed: {e}")
                self.errors += 1

                # Fall through to template if fallback enabled
                if not self.fallback_to_template:
                    raise

        # Fallback to template
        logger.info("Using template fallback for response generation")
        response = self.template_provider.generate(prompt, max_tokens, temperature)
        latency_ms = (time.time() - start_time) * 1000

        self.template_requests += 1

        metadata = {
            "provider": "template",
            "model": "none",
            "latency_ms": round(latency_ms, 2),
            "tokens_estimated": 0,
            "cached": False,
            "success": True,
            "fallback_reason": "rate_limit_or_disabled" if self.enabled else "llm_disabled"
        }

        return response, metadata

    def get_statistics(self) -> Dict:
        """Get usage statistics."""
        github_rate_limit_used = 0
        if self.github_provider and hasattr(self.github_provider, 'request_count'):
            github_rate_limit_used = self.github_provider.request_count

        return {
            "total_requests": self.total_requests,
            "github_requests": self.github_requests,
            "template_requests": self.template_requests,
            "cache_hits": self.cache_hits,
            "errors": self.errors,
            "github_daily_limit": self.github_provider.rate_limit_per_day if self.github_provider else 0,
            "github_requests_today": github_rate_limit_used,
            "github_remaining_today": max(0, (self.github_provider.rate_limit_per_day - github_rate_limit_used)) if self.github_provider else 0,
        }
