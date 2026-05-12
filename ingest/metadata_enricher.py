"""
Metadata enrichment module for GenAI RAG System.
Auto-tags documents with department, topic, date, and author using LLM.
"""

import json
import re
from typing import Optional

from config.settings import get_settings
from schemas.documents import Document
from utils.logger import get_logger

logger = get_logger(__name__)

_ENRICHMENT_PROMPT = """Analyze the following document excerpt and return a JSON object with these fields:
- "department": The department this document is most relevant to (e.g., Engineering, Legal, HR, Finance, Marketing, Operations, General).
- "topic": A concise topic label (2-5 words).
- "author": The author name if identifiable, otherwise null.
- "date_mentioned": The most prominent date mentioned (ISO format YYYY-MM-DD) if any, otherwise null.
- "language": The language of the document (e.g., English, Spanish).
- "document_type": The type of document (e.g., tutorial, policy, report, guide, reference, faq).

Document excerpt (first 1500 chars):
\"\"\"
{excerpt}
\"\"\"

Return ONLY valid JSON, no markdown fences."""


class MetadataEnricher:
    """Enriches documents with auto-generated metadata using LLM."""

    def __init__(self):
        self.settings = get_settings()
        self._llm_client = None

    def _get_client(self):
        """Lazy-init LLM client (avoids import cycle with pipeline)."""
        if self._llm_client is None:
            from pipeline.llm_service import LLMService
            self._llm_client = LLMService()
        return self._llm_client

    def enrich(self, document: Document) -> Document:
        """
        Enrich a document's metadata with LLM-generated tags.
        Adds keys to document.metadata.custom_metadata dict.
        """
        if not self.settings.metadata_enrichment_enabled:
            return document

        excerpt = document.content[:1500]
        prompt = _ENRICHMENT_PROMPT.format(excerpt=excerpt)

        try:
            llm = self._get_client()
            raw = llm.generate(prompt, system_message="You are a document classifier. Return only JSON.")

            # Parse JSON from response (strip markdown fences if present)
            raw = raw.strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json)?\s*", "", raw)
                raw = re.sub(r"\s*```$", "", raw)

            tags = json.loads(raw)

            # Merge into custom_metadata
            if document.metadata.custom_metadata is None:
                document.metadata.custom_metadata = {}
            document.metadata.custom_metadata.update({
                "department": tags.get("department", "General"),
                "topic": tags.get("topic"),
                "author": tags.get("author"),
                "date_mentioned": tags.get("date_mentioned"),
                "language": tags.get("language", "English"),
                "document_type": tags.get("document_type"),
            })

            logger.info(
                f"Enriched metadata for {document.metadata.source}: "
                f"dept={tags.get('department')}, topic={tags.get('topic')}"
            )

        except Exception as e:
            logger.warning(f"Metadata enrichment failed for {document.metadata.source}: {e}")

        return document

    def enrich_batch(self, documents: list[Document]) -> list[Document]:
        """Enrich a list of documents."""
        if not self.settings.metadata_enrichment_enabled:
            return documents
        return [self.enrich(doc) for doc in documents]
