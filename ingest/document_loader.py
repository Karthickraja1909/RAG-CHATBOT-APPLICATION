

"""
Document loader module for GenAI RAG System.
Handles loading documents from various formats (PDF, TXT, MD, DOCX, CSV).
"""

import csv
import hashlib
from io import StringIO
from pathlib import Path
from typing import Union

from config.settings import get_settings
from schemas.documents import Document, DocumentMetadata
from utils.exceptions import DocumentIngestionError
from utils.logger import get_logger

logger = get_logger(__name__)


class DocumentLoader:
    """Loads documents from various file formats."""

    def __init__(self):
        self.settings = get_settings()
        self._loaders = {
            ".txt": self._load_text,
            ".md": self._load_text,
            ".csv": self._load_csv,
            ".pdf": self._load_pdf,
            ".docx": self._load_docx,
        }

    def load_document(self, file_path: Union[str, Path]) -> Document:
        """
        Load a single document from file.

        Args:
            file_path: Path to the document file.

        Returns:
            Document object with content and metadata.

        Raises:
            DocumentIngestionError: If file cannot be loaded.
        """
        path = Path(file_path)
        self._validate_file(path)

        extension = path.suffix.lower()
        loader_fn = self._loaders.get(extension)

        if loader_fn is None:
            raise DocumentIngestionError(
                f"No loader available for file type: {extension}",
                details={"file_path": str(path), "supported": list(self._loaders.keys())},
            )

        try:
            content = loader_fn(path)
            doc_id = self._generate_document_id(path)
            metadata = DocumentMetadata(
                source=str(path),
                file_type=extension,
                file_size_bytes=path.stat().st_size,
            )

            document = Document(document_id=doc_id, content=content, metadata=metadata)
            logger.info(f"Loaded document: {path.name} ({len(content)} chars)")
            return document

        except DocumentIngestionError:
            raise
        except Exception as e:
            raise DocumentIngestionError(
                f"Failed to load document: {path}",
                details={"error": str(e), "file_path": str(path)},
            )

    def load_directory(self, directory: Union[str, Path] = None) -> list[Document]:
        """
        Load all supported documents from a directory.

        Args:
            directory: Directory path. Defaults to configured documents directory.

        Returns:
            List of Document objects.
        """
        dir_path = Path(directory) if directory else Path(self.settings.documents_directory)

        if not dir_path.exists():
            logger.warning(f"Directory does not exist: {dir_path}")
            return []

        documents = []
        supported = self.settings.supported_extensions

        for file_path in sorted(dir_path.rglob("*")):
            if file_path.is_file() and file_path.suffix.lower() in supported:
                try:
                    doc = self.load_document(file_path)
                    documents.append(doc)
                except DocumentIngestionError as e:
                    logger.error(f"Skipping file {file_path}: {e.message}")

        logger.info(f"Loaded {len(documents)} documents from {dir_path}")
        return documents

    def _load_text(self, path: Path) -> str:
        """Load plain text or markdown file."""
        return path.read_text(encoding="utf-8")

    def _load_csv(self, path: Path) -> str:
        """Load CSV file and convert to text representation."""
        content_parts = []
        raw_text = path.read_text(encoding="utf-8")
        reader = csv.DictReader(StringIO(raw_text))

        for row_num, row in enumerate(reader, 1):
            row_text = " | ".join(f"{k}: {v}" for k, v in row.items() if v)
            content_parts.append(f"Row {row_num}: {row_text}")

        return "\n".join(content_parts)

    def _load_pdf(self, path: Path) -> str:
        """Load PDF file content."""
        try:
            import PyPDF2

            content_parts = []
            with open(path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page_num, page in enumerate(reader.pages, 1):
                    text = page.extract_text()
                    if text.strip():
                        content_parts.append(f"[Page {page_num}]\n{text}")

            return "\n\n".join(content_parts)

        except ImportError:
            raise DocumentIngestionError(
                "PyPDF2 is required for PDF loading. Install with: pip install PyPDF2",
                details={"file_path": str(path)},
            )

    def _load_docx(self, path: Path) -> str:
        """Load DOCX file content."""
        try:
            import docx

            doc = docx.Document(str(path))
            paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
            return "\n\n".join(paragraphs)

        except ImportError:
            raise DocumentIngestionError(
                "python-docx is required for DOCX loading. Install with: pip install python-docx",
                details={"file_path": str(path)},
            )

    def _validate_file(self, path: Path) -> None:
        """Validate file exists, is within size limits, and is supported."""
        if not path.exists():
            raise DocumentIngestionError(f"File not found: {path}")
        if not path.is_file():
            raise DocumentIngestionError(f"Path is not a file: {path}")

        max_bytes = self.settings.max_file_size_mb * 1024 * 1024
        if path.stat().st_size > max_bytes:
            raise DocumentIngestionError(
                f"File exceeds max size ({self.settings.max_file_size_mb}MB): {path}"
            )

        if path.suffix.lower() not in self.settings.supported_extensions:
            raise DocumentIngestionError(f"Unsupported file type: {path.suffix}")

    def _generate_document_id(self, path: Path) -> str:
        """Generate a deterministic document ID from file path and modification time."""
        key = f"{path.resolve()}:{path.stat().st_mtime}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]