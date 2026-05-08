"""
Ingestion pipeline module for GenAI RAG System.
Orchestrates the full document loading, splitting, and embedding process.
"""

import time
from pathlib import Path
from typing import Optional, Union

from config.settings import get_settings
from ingest.document_loader import DocumentLoader
from ingest.text_splitter import TextSplitter
from schemas.documents import Document, DocumentChunk
from utils.exceptions import DocumentIngestionError
from utils.logger import get_logger

logger = get_logger(__name__)


class IngestionPipeline:
    """Orchestrates document ingestion: load → split → embed → store."""

    def __init__(
        self,
        document_loader: Optional[DocumentLoader] = None,
        text_splitter: Optional[TextSplitter] = None,
        vector_store=None,
    ):
        self.settings = get_settings()
        self.document_loader = document_loader or DocumentLoader()
        self.text_splitter = text_splitter or TextSplitter()
        self.vector_store = vector_store

    def ingest_file(self, file_path: Union[str, Path]) -> list[DocumentChunk]:
        """
        Ingest a single file: load, split, and optionally store in vector DB.

        Args:
            file_path: Path to the document file.

        Returns:
            List of document chunks created.
        """
        start_time = time.time()
        logger.info(f"Ingesting file: {file_path}")

        document = self.document_loader.load_document(file_path)
        chunks = self.text_splitter.split_document(document)

        if self.vector_store and chunks:
            self.vector_store.add_chunks(chunks)
            logger.info(f"Stored {len(chunks)} chunks in vector store")

        elapsed = time.time() - start_time
        logger.info(f"Ingestion complete: {len(chunks)} chunks in {elapsed:.2f}s")
        return chunks

    def ingest_directory(self, directory: Union[str, Path] = None) -> list[DocumentChunk]:
        """
        Ingest all documents from a directory.

        Args:
            directory: Directory path. Defaults to configured documents directory.

        Returns:
            List of all document chunks created.
        """
        dir_path = directory or self.settings.documents_directory
        start_time = time.time()
        logger.info(f"Ingesting directory: {dir_path}")

        documents = self.document_loader.load_directory(dir_path)

        if not documents:
            logger.warning(f"No documents found in: {dir_path}")
            return []

        all_chunks = self.text_splitter.split_documents(documents)

        if self.vector_store and all_chunks:
            self.vector_store.add_chunks(all_chunks)
            logger.info(f"Stored {len(all_chunks)} chunks in vector store")

        elapsed = time.time() - start_time
        logger.info(
            f"Directory ingestion complete: {len(documents)} docs → {len(all_chunks)} chunks in {elapsed:.2f}s"
        )
        return all_chunks

    def ingest_documents(self, documents: list[Document]) -> list[DocumentChunk]:
        """
        Ingest pre-loaded documents (useful for testing/custom sources).

        Args:
            documents: List of Document objects.

        Returns:
            List of document chunks.
        """
        all_chunks = self.text_splitter.split_documents(documents)

        if self.vector_store and all_chunks:
            self.vector_store.add_chunks(all_chunks)

        return all_chunks