"""
LLM-based reranker for reducing retrieval noise in the RAG pipeline.

Scores each retrieved chunk for query relevance using an LLM judge, then filters
and re-orders by relevance score. This dramatically improves ContextualPrecision
by ensuring only truly relevant chunks reach the generation step.

All configuration via .env:
  RERANKING_ENABLED=true
  RERANKER_MODEL=gpt-4o-mini
  RERANKER_TOP_N=4
  RERANKER_RELEVANCE_THRESHOLD=0.5
"""

import json
from typing import Optional

import openai

from config.settings import get_settings
from schemas.pipeline import RetrievalResult
from utils.exceptions import PipelineError
from utils.logger import get_logger

logger = get_logger(__name__)

_RERANK_PROMPT = """You are a relevance scoring engine. Given a user query and a document chunk, rate how relevant the chunk is for answering the query.

Score from 0.0 to 1.0:
  1.0 = directly answers the query
  0.7 = contains key information needed to answer
  0.4 = tangentially related but not directly useful
  0.1 = not relevant at all

USER QUERY:
{query}

DOCUMENT CHUNK:
{chunk}

Respond with ONLY a JSON object, no markdown:
{{"score": <float>, "reason": "<one sentence explanation>"}}"""


class Reranker:
    """
    LLM-based reranker that scores retrieved chunks for query relevance.

    Flow: Retrieved chunks → LLM scores each for relevance → Filter by threshold → Sort → Top-N
    """

    def __init__(
        self,
        model: Optional[str] = None,
        top_n: Optional[int] = None,
        relevance_threshold: Optional[float] = None,
    ):
        self.settings = get_settings()
        self.model = model or self.settings.reranker_model
        self.top_n = top_n or self.settings.reranker_top_n
        self.relevance_threshold = relevance_threshold or self.settings.reranker_relevance_threshold
        self._client = self._create_client()

    def _create_client(self) -> openai.OpenAI:
        """Create OpenAI client matching the configured provider."""
        if self.settings.use_openrouter and self.settings.openrouter_api_key:
            self.model = self.settings.openrouter_model
            return openai.OpenAI(
                api_key=self.settings.openrouter_api_key,
                base_url=self.settings.openrouter_base_url,
                timeout=30.0,
            )
        elif self.settings.azure_openai_endpoint:
            self.model = self.settings.azure_openai_deployment or self.model
            return openai.AzureOpenAI(
                api_key=self.settings.azure_openai_api_key,
                api_version=self.settings.azure_openai_api_version,
                azure_endpoint=self.settings.azure_openai_endpoint,
                timeout=30.0,
            )
        else:
            return openai.OpenAI(
                api_key=self.settings.openai_api_key,
                base_url=self.settings.openai_api_base,
                timeout=30.0,
            )

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """
        Rerank retrieved chunks by LLM-judged relevance to the query.

        1. Score each chunk with the LLM
        2. Drop chunks below relevance_threshold
        3. Sort by relevance score descending
        4. Return top_n results

        Args:
            query: The user's original query.
            results: Retrieved chunks from vector search.

        Returns:
            Filtered, re-ordered list of RetrievalResult.
        """
        if not results:
            return results

        if len(results) <= 1:
            return results

        logger.info(
            f"Reranking {len(results)} chunks (model={self.model}, "
            f"threshold={self.relevance_threshold}, top_n={self.top_n})"
        )

        scored_results = []

        for result in results:
            try:
                relevance_score, reason = self._score_chunk(query, result.content)

                scored_results.append({
                    "result": result,
                    "relevance_score": relevance_score,
                    "reason": reason,
                })

                logger.debug(
                    f"  [{result.chunk_id}] sim={result.score:.3f} → relevance={relevance_score:.2f} "
                    f"({'KEEP' if relevance_score >= self.relevance_threshold else 'DROP'}) "
                    f"{reason}"
                )

            except Exception as e:
                # On scoring failure, keep the chunk with its original similarity score
                logger.warning(f"  Reranking failed for {result.chunk_id}: {e}. Keeping with original score.")
                scored_results.append({
                    "result": result,
                    "relevance_score": result.score,
                    "reason": "scoring_failed",
                })

        # Filter by threshold
        filtered = [
            sr for sr in scored_results
            if sr["relevance_score"] >= self.relevance_threshold
        ]

        dropped_count = len(scored_results) - len(filtered)
        if dropped_count > 0:
            logger.info(f"Reranker dropped {dropped_count} irrelevant chunks (below {self.relevance_threshold})")

        if not filtered:
            # If all chunks were dropped, keep the best one to avoid empty context
            logger.warning("All chunks below threshold. Keeping highest-scored chunk.")
            best = max(scored_results, key=lambda x: x["relevance_score"])
            filtered = [best]

        # Sort by relevance score descending
        filtered.sort(key=lambda x: x["relevance_score"], reverse=True)

        # Trim to top_n
        top_results = [sr["result"] for sr in filtered[:self.top_n]]

        logger.info(
            f"Reranking complete: {len(results)} → {len(top_results)} chunks "
            f"(best_relevance={filtered[0]['relevance_score']:.2f})"
        )

        return top_results

    def _score_chunk(self, query: str, chunk_content: str) -> tuple[float, str]:
        """
        Score a single chunk's relevance to the query using LLM.

        Returns:
            Tuple of (relevance_score, reason).
        """
        prompt = _RERANK_PROMPT.format(
            query=query,
            chunk=chunk_content[:1500],  # Cap to avoid token limits
        )

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a relevance scoring engine. Output ONLY valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=100,
            )

            text = response.choices[0].message.content.strip()

            # Handle markdown fences
            if text.startswith("```"):
                text = text.split("\n", 1)[1]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()

            parsed = json.loads(text)
            score = float(parsed.get("score", 0.0))
            reason = str(parsed.get("reason", ""))

            # Clamp score to valid range
            score = max(0.0, min(1.0, score))
            return score, reason

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            raise PipelineError(
                f"Failed to parse reranker response: {e}",
                details={"query": query[:100], "chunk": chunk_content[:100]},
            )
        except openai.APIError as e:
            raise PipelineError(
                f"Reranker LLM call failed: {e}",
                details={"model": self.model},
            )
