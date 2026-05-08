"""
File handling utilities for GenAI RAG System.
Supports reading, writing, and validating various file formats.
"""

import json
from pathlib import Path
from typing import Any, Union

from config.settings import get_settings
from utils.exceptions import DocumentIngestionError
from utils.logger import get_logger

logger = get_logger(__name__)


class FileHandler:
    """Utility class for file operations."""

    def __init__(self):
        self.settings = get_settings()

    def read_text_file(self, file_path: Union[str, Path]) -> str:
        """Read and return contents of a text file."""
        path = Path(file_path)
        self._validate_file_exists(path)
        self._validate_file_size(path)

        try:
            return path.read_text(encoding="utf-8")
        except Exception as e:
            raise DocumentIngestionError(
                f"Failed to read file: {path}",
                details={"error": str(e), "file_path": str(path)},
            )

    def read_json_file(self, file_path: Union[str, Path]) -> Any:
        """Read and parse a JSON file."""
        content = self.read_text_file(file_path)
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise DocumentIngestionError(
                f"Invalid JSON in file: {file_path}",
                details={"error": str(e), "file_path": str(file_path)},
            )

    def write_json_file(self, file_path: Union[str, Path], data: Any) -> None:
        """Write data to a JSON file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
            logger.info(f"Written JSON file: {path}")
        except Exception as e:
            raise DocumentIngestionError(
                f"Failed to write file: {path}",
                details={"error": str(e)},
            )

    def write_text_file(self, file_path: Union[str, Path], content: str) -> None:
        """Write text content to a file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            path.write_text(content, encoding="utf-8")
            logger.info(f"Written text file: {path}")
        except Exception as e:
            raise DocumentIngestionError(
                f"Failed to write file: {path}",
                details={"error": str(e)},
            )

    def list_documents(self, directory: Union[str, Path] = None) -> list[Path]:
        """List all supported documents in a directory."""
        dir_path = Path(directory) if directory else Path(self.settings.documents_directory)

        if not dir_path.exists():
            logger.warning(f"Documents directory does not exist: {dir_path}")
            return []

        supported = self.settings.supported_extensions
        files = []
        for ext in supported:
            files.extend(dir_path.rglob(f"*{ext}"))

        logger.info(f"Found {len(files)} documents in {dir_path}")
        return sorted(files)

    def ensure_directory(self, directory: Union[str, Path]) -> Path:
        """Create directory if it doesn't exist and return the Path."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _validate_file_exists(self, path: Path) -> None:
        """Validate that a file exists."""
        if not path.exists():
            raise DocumentIngestionError(
                f"File not found: {path}",
                details={"file_path": str(path)},
            )
        if not path.is_file():
            raise DocumentIngestionError(
                f"Path is not a file: {path}",
                details={"file_path": str(path)},
            )

    def _validate_file_size(self, path: Path) -> None:
        """Validate file size is within limits."""
        max_bytes = self.settings.max_file_size_mb * 1024 * 1024
        file_size = path.stat().st_size

        if file_size > max_bytes:
            raise DocumentIngestionError(
                f"File exceeds maximum size ({self.settings.max_file_size_mb}MB): {path}",
                details={
                    "file_path": str(path),
                    "file_size_mb": round(file_size / (1024 * 1024), 2),
                    "max_size_mb": self.settings.max_file_size_mb,
                },
            )

    def _validate_extension(self, path: Path) -> None:
        """Validate file extension is supported."""
        if path.suffix.lower() not in self.settings.supported_extensions:
            raise DocumentIngestionError(
                f"Unsupported file type: {path.suffix}",
                details={
                    "file_path": str(path),
                    "extension": path.suffix,
                    "supported": self.settings.supported_extensions,
                },
            )