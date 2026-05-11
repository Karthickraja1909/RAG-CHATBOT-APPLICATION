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

    def load_directory(self, directory: Union[str, Path, None] = None) -> list[Document]:
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
        """Load PDF file content with structure preservation."""
        try:
            import PyPDF2

            content_parts = []
            with open(path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page_num, page in enumerate(reader.pages, 1):
                    text = page.extract_text()
                    if not text or not text.strip():
                        continue
                    # Clean up common PDF extraction artifacts
                    lines = text.split("\n")
                    cleaned_lines = []
                    for line in lines:
                        stripped = line.strip()
                        if not stripped:
                            cleaned_lines.append("")
                            continue
                        # Detect likely headings: short lines in ALL CAPS or Title Case
                        # that don't end with punctuation
                        if (
                            len(stripped) < 100
                            and not stripped.endswith((".", ",", ";", ":"))
                            and (stripped.isupper() or stripped.istitle())
                        ):
                            cleaned_lines.append(f"\n## {stripped}")
                        else:
                            cleaned_lines.append(stripped)
                    page_text = "\n".join(cleaned_lines)
                    content_parts.append(f"[Page {page_num}]\n{page_text}")

            return "\n\n".join(content_parts)

        except ImportError:
            raise DocumentIngestionError(
                "PyPDF2 is required for PDF loading. Install with: pip install PyPDF2",
                details={"file_path": str(path)},
            )

    def _load_docx(self, path: Path) -> str:
        """Load DOCX file content with headings, tables, and structure preserved."""
        try:
            import docx
            from docx.table import Table as DocxTable
            from docx.text.paragraph import Paragraph

            doc = docx.Document(str(path))
            content_parts = []

            for element in doc.element.body:
                tag = element.tag.split("}")[-1] if "}" in element.tag else element.tag

                if tag == "p":
                    para = Paragraph(element, doc)
                    text = para.text.strip()
                    if not text:
                        continue
                    style_name = ""
                    if para.style is not None and para.style.name is not None:
                        style_name = para.style.name.lower()
                    # Convert headings to markdown-style headers
                    if "heading 1" in style_name or "title" in style_name:
                        content_parts.append(f"\n# {text}\n")
                    elif "heading 2" in style_name:
                        content_parts.append(f"\n## {text}\n")
                    elif "heading 3" in style_name:
                        content_parts.append(f"\n### {text}\n")
                    elif "heading" in style_name:
                        content_parts.append(f"\n#### {text}\n")
                    elif "toc" in style_name:
                        continue  # Skip table of contents entries
                    else:
                        content_parts.append(text)

                elif tag == "tbl":
                    table = DocxTable(element, doc)
                    table_text = self._extract_table(table)
                    if table_text:
                        content_parts.append(table_text)

            return "\n\n".join(content_parts)

        except ImportError:
            raise DocumentIngestionError(
                "python-docx is required for DOCX loading. Install with: pip install python-docx",
                details={"file_path": str(path)},
            )

    def _extract_table(self, table) -> str:
        """Extract table content as readable markdown-style text."""
        rows = []
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            if any(cells):
                rows.append(cells)

        if not rows:
            return ""

        # First row as header
        header = rows[0]
        lines = [" | ".join(header)]
        lines.append(" | ".join("---" for _ in header))
        for row in rows[1:]:
            # Pad row to header length
            padded = row + [""] * (len(header) - len(row))
            lines.append(" | ".join(padded[:len(header)]))

        return "\n".join(lines)

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