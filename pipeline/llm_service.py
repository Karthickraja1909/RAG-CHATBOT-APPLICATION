

"""
LLM service module for GenAI RAG System.
Handles all interactions with the language model (OpenAI/Azure OpenAI).
"""

from typing import Optional

import openai

from config.settings import get_settings
from utils.exceptions import LLMError
from utils.logger import get_logger

logger = get_logger(__name__)


class LLMService:
    """Service for interacting with LLM APIs."""

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

        # Provider selection: OpenRouter > Azure > OpenAI
        if self.settings.use_openrouter and self.settings.openrouter_api_key:
            self.client = openai.OpenAI(
                api_key=self.settings.openrouter_api_key,
                base_url=self.settings.openrouter_base_url,
            )
            self.model = self.settings.openrouter_model
            logger.info(f"LLM Service using OpenRouter: {self.model}")
        elif self.settings.azure_openai_endpoint:
            self.client = openai.AzureOpenAI(
                api_key=self.settings.azure_openai_api_key,
                api_version=self.settings.azure_openai_api_version,
                azure_endpoint=self.settings.azure_openai_endpoint,
            )
            self.model = self.settings.azure_openai_deployment or self.model
            logger.info(f"LLM Service using Azure OpenAI: {self.model}")
        else:
            self.client = openai.OpenAI(
                api_key=self.settings.openai_api_key,
                base_url=self.settings.openai_api_base,
            )
            logger.info(f"LLM Service using OpenAI: {self.model}")

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

            answer = response.choices[0].message.content
            logger.debug(
                f"LLM response generated (model={self.model}, "
                f"tokens={response.usage.total_tokens if response.usage else 'N/A'})"
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
        Generate a response using a context-aware prompt template.

        Args:
            query: User's question.
            context: Retrieved context documents.
            system_prompt_template: Template with {context} and {question} placeholders.

        Returns:
            Generated response.
        """
        prompt = system_prompt_template.format(context=context, question=query)
        return self.generate(prompt)