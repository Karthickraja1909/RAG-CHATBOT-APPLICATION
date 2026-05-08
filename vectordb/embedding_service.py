


"""
Embedding service module for GenAI RAG System.
Handles generating vector embeddings using OpenAI or Azure OpenAI.
"""

import time
from typing import Optional

import openai

from config.settings import get_settings
from utils.exceptions import LLMError
from utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    """Service for generating text embeddings."""

    def __init__(self, model: Optional[str] = None):
        self.settings = get_settings()
        self.model = model or self.settings.openai_embedding_model

        # Provider selection: OpenRouter > Azure > OpenAI
        if self.settings.use_openrouter and self.settings.openrouter_api_key:
            self.client = openai.OpenAI(
                api_key=self.settings.openrouter_api_key,
                base_url=self.settings.openrouter_base_url,
            )
            # Use OpenRouter-specific embedding model if no explicit model passed
            if not model:
                self.model = self.settings.openrouter_embedding_model
            logger.info(f"Embedding Service using OpenRouter: {self.model}")
        elif self.settings.azure_openai_endpoint:
            self.client = openai.AzureOpenAI(
                api_key=self.settings.azure_openai_api_key,
                api_version=self.settings.azure_openai_api_version,
                azure_endpoint=self.settings.azure_openai_endpoint,
            )
            logger.info(f"Embedding Service using Azure OpenAI: {self.model}")
        else:
            self.client = openai.OpenAI(
                api_key=self.settings.openai_api_key,
                base_url=self.settings.openai_api_base,
            )
            logger.info(f"Embedding Service using OpenAI: {self.model}")

    def embed_text(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed.

        Returns:
            List of floats representing the embedding vector.
        """
        if not text.strip():
            raise LLMError("Cannot embed empty text")

        try:
            response = self.client.embeddings.create(
                input=[text],
                model=self.model,
            )
            return response.data[0].embedding

        except openai.APIError as e:
            raise LLMError(
                f"OpenAI API error during embedding: {e}",
                details={"model": self.model, "text_length": len(text)},
            )

    def embed_texts(self, texts: list[str], batch_size: int = 100) -> list[list[float]]:
        """
        Generate embeddings for multiple texts with batching.

        Args:
            texts: List of texts to embed.
            batch_size: Number of texts per API call.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        all_embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        start_time = time.time()

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_num = (i // batch_size) + 1

            try:
                # Filter out empty texts
                valid_texts = [t if t.strip() else " " for t in batch]

                response = self.client.embeddings.create(
                    input=valid_texts,
                    model=self.model,
                )

                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)

                logger.debug(f"Embedded batch {batch_num}/{total_batches} ({len(batch)} texts)")

            except openai.APIError as e:
                raise LLMError(
                    f"OpenAI API error in batch {batch_num}: {e}",
                    details={"batch_num": batch_num, "batch_size": len(batch)},
                )

        elapsed = time.time() - start_time
        logger.info(f"Generated {len(all_embeddings)} embeddings in {elapsed:.2f}s")
        return all_embeddings