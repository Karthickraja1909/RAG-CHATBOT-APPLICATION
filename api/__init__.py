"""
FastAPI REST API package for GenAI RAG System.

Run: uvicorn api.app:app --host 0.0.0.0 --port 8000
Or:  python -m api.app
"""

from api.app import app  # noqa: F401

__all__ = ["app"]
