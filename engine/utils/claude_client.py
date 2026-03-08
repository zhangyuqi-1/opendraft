"""
ABOUTME: Claude (Anthropic) client wrapper.
ABOUTME: Provides same generate_content() interface as GeminiModelWrapper.
"""

import os
import time
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Transient network errors that warrant a retry
_RETRYABLE = (
    "incomplete chunked read",
    "peer closed connection",
    "RemoteProtocolError",
    "ConnectError",
    "ReadTimeout",
    "ConnectionError",
)


class ClaudeResponse:
    """Wraps Anthropic response to expose a .text attribute."""

    def __init__(self, content_blocks: list, usage: Any = None):
        texts = [b.text for b in content_blocks if hasattr(b, "text")]
        self.text = "".join(texts)
        self.usage_metadata = usage


class ClaudeModelWrapper:
    """
    Wrapper around the Anthropic client that exposes the same
    generate_content() interface used throughout agent_runner.py.
    """

    def __init__(self, client: Any, model_name: str, temperature: float = 0.7):
        self.client = client
        self.model_name = model_name
        self.default_temperature = temperature

    def generate_content(
        self,
        prompt: Any,
        generation_config: Any = None,
        safety_settings: Any = None,
    ) -> ClaudeResponse:
        _ = safety_settings

        temperature = self.default_temperature
        max_tokens = 8192

        if generation_config:
            if hasattr(generation_config, "temperature"):
                temperature = generation_config.temperature
            if hasattr(generation_config, "max_output_tokens") and generation_config.max_output_tokens:
                max_tokens = generation_config.max_output_tokens
            if isinstance(generation_config, dict):
                temperature = generation_config.get("temperature", temperature)
                max_tokens = generation_config.get("max_output_tokens", max_tokens)

        if isinstance(prompt, list):
            prompt = "\n".join(str(p) for p in prompt)
        else:
            prompt = str(prompt)

        max_retries = 5
        for attempt in range(max_retries):
            try:
                return self._stream_once(prompt, max_tokens, temperature)
            except Exception as e:
                err = str(e)
                is_retryable = any(kw in err for kw in _RETRYABLE)
                if is_retryable and attempt < max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning(
                        f"[ClaudeClient] Stream interrupted (attempt {attempt+1}/{max_retries}): {err!r}. "
                        f"Retrying in {wait}s..."
                    )
                    time.sleep(wait)
                else:
                    raise

    def _stream_once(self, prompt: str, max_tokens: int, temperature: float) -> ClaudeResponse:
        """Single streaming attempt; raises on any network error."""
        from anthropic.types import TextBlock

        text_parts = []
        usage = None
        with self.client.messages.stream(
            model=self.model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            for text in stream.text_stream:
                text_parts.append(text)
            final = stream.get_final_message()
            usage = final.usage

        content = [TextBlock(type="text", text="".join(text_parts), citations=None)]
        return ClaudeResponse(content, usage)


def create_claude_client(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model_name: str = "claude-sonnet-4-6",
    temperature: float = 0.7,
) -> ClaudeModelWrapper:
    """
    Create a Claude client wrapper.

    Args:
        api_key: Anthropic API key (falls back to ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN env vars)
        base_url: Custom base URL (falls back to ANTHROPIC_BASE_URL env var)
        model_name: Model to use
        temperature: Default temperature

    Returns:
        ClaudeModelWrapper instance
    """
    try:
        import anthropic
    except ImportError:
        raise ImportError("anthropic not installed. Run: pip install anthropic")

    api_key = (
        api_key
        or os.getenv("ANTHROPIC_API_KEY")
        or os.getenv("ANTHROPIC_AUTH_TOKEN")
    )
    if not api_key:
        raise ValueError(
            "API key required. Set ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN."
        )

    base_url = base_url or os.getenv("ANTHROPIC_BASE_URL")

    kwargs: dict = {
        "api_key": api_key,
        "default_headers": {"User-Agent": "curl/7.88.1"},
    }
    if base_url:
        kwargs["base_url"] = base_url

    client = anthropic.Anthropic(**kwargs)
    return ClaudeModelWrapper(client, model_name, temperature)
