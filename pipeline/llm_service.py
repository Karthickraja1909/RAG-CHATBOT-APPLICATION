"""
LLM service module for GenAI RAG System.
Handles all interactions with the language model (OpenAI/Azure OpenAI).
Features: connection pooling, token usage tracking per request & cumulative.
"""

import threading
from typing import Optional

import httpx
import openai

from config.settings import get_settings
from utils.exceptions import LLMError
from utils.logger import get_logger

logger = get_logger(__name__)


class TokenUsage:
    """Thread-safe cumulative token usage tracker."""

    def __init__(self):
        self._lock = threading.Lock()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.request_count = 0

    def record(self, usage) -> dict:
        """Record token usage from an API response. Returns per-request dict."""
        if usage is None:
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        per_request = {
            "prompt_tokens": usage.prompt_tokens or 0,
            "completion_tokens": usage.completion_tokens or 0,
            "total_tokens": usage.total_tokens or 0,
        }
        with self._lock:
            self.prompt_tokens += per_request["prompt_tokens"]
            self.completion_tokens += per_request["completion_tokens"]
            self.total_tokens += per_request["total_tokens"]
            self.request_count += 1
        return per_request

    @property
    def summary(self) -> dict:
        with self._lock:
            return {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
                "request_count": self.request_count,
            }

    def reset(self) -> None:
        with self._lock:
            self.prompt_tokens = 0
            self.completion_tokens = 0
            self.total_tokens = 0
            self.request_count = 0


class LLMService:
    """Service for interacting with LLM APIs with connection pooling & token tracking."""

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        self.settings = get_settings()
        self.model = model or self.settings.openai_model
        self.temperature = temperature if temperature is not None else self.settings.openai_temperature
        self.max_tokens = max_tokens or self.settings.openai_max_tokens

        # Token usage tracker
        self.token_usage = TokenUsage()

        # Connection pool via httpx (reuse TCP connections across requests)
        pool_size = self.settings.llm_connection_pool_size
        timeout = self.settings.llm_request_timeout
        http_client = httpx.Client(
            limits=httpx.Limits(
                max_connections=pool_size,
                max_keepalive_connections=pool_size,
            ),
            timeout=httpx.Timeout(timeout, connect=10.0),
        )

        # Provider selection: OpenRouter > Azure > OpenAI
        if self.settings.use_openrouter and self.settings.openrouter_api_key:
            self.client = openai.OpenAI(
                api_key=self.settings.openrouter_api_key,
                base_url=self.settings.openrouter_base_url,
                http_client=http_client,
            )
            self.model = self.settings.openrouter_model
            logger.info(f"LLM Service using OpenRouter: {self.model} (pool={pool_size})")
        elif self.settings.azure_openai_endpoint:
            self.client = openai.AzureOpenAI(
                api_key=self.settings.azure_openai_api_key,
                api_version=self.settings.azure_openai_api_version,
                azure_endpoint=self.settings.azure_openai_endpoint,
                http_client=http_client,
            )
            self.model = self.settings.azure_openai_deployment or self.model
            logger.info(f"LLM Service using Azure OpenAI: {self.model} (pool={pool_size})")
        else:
            self.client = openai.OpenAI(
                api_key=self.settings.openai_api_key,
                base_url=self.settings.openai_api_base,
                http_client=http_client,
            )
            logger.info(f"LLM Service using OpenAI: {self.model} (pool={pool_size})")

    def generate(
        self,
        prompt: str,
        system_message: Optional[str] = None,
    ) -> str:
        """
        Generate a response from the LLM.

        Args:
            prompt: User prompt/question.
            system_message: Optional system message for context.

        Returns:
            Generated text response.
        """
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            # Track token usage
            per_request = self.token_usage.record(response.usage)

            answer = response.choices[0].message.content
            logger.debug(
                f"LLM response generated (model={self.model}, "
                f"tokens={per_request['total_tokens']})"
            )
            return answer

        except openai.APIError as e:
            raise LLMError(
                f"LLM generation failed: {e}",
                details={"model": self.model, "prompt_length": len(prompt)},
            )

    def generate_with_context(
        self,
        query: str,
        context: str,
        system_prompt_template: str,
    ) -> str:
        """
        Generate a response using retrieved context.
        Sends context as system message and query as user message
        for better LLM instruction following.
        """
        # Build system message: inject context into template, remove the question placeholder
        system_message = system_prompt_template.format(context=context, question="").strip()
        # Remove trailing "User Question:" or "User Question: " if left empty
        for trailing in ["User Question:", "User Question: ", "Answer:", "Answer: "]:
            if system_message.endswith(trailing):
                system_message = system_message[: -len(trailing)].strip()

        return self.generate(query, system_message=system_message)
