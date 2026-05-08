"""
Document ingestion script.
Loads documents from the configured directory and builds the FAISS vector index.

Usage:
    python -m scripts.ingest_documents --directory data/documents
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import get_settings
from ingest.ingestion_pipeline import IngestionPipeline
from vectordb.faiss_store import FAISSVectorStore
from utils.logger import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Ingest documents into vector store")

    parser.add_argument(
        "--directory",
        type=str,
        default=None,
        help="Directory containing documents to ingest",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Single file to ingest",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing vector store before ingestion",
    )

    return parser.parse_args()


def main():
    """Main ingestion entry point."""
    args = parse_args()
    settings = get_settings()

    logger.info("=" * 50)
    logger.info("GenAI RAG System - Document Ingestion")
    logger.info("=" * 50)

    # Initialize vector store
    vector_store = FAISSVectorStore()

    if args.clear:
        logger.info("Clearing existing vector store...")
        vector_store.clear()

    # Initialize pipeline
    pipeline = IngestionPipeline(vector_store=vector_store)

    try:
        if args.file:
            chunks = pipeline.ingest_file(args.file)
        else:
            directory = args.directory or settings.documents_directory
            chunks = pipeline.ingest_directory(directory)

        # Print statistics
        stats = vector_store.get_stats()
        logger.info("\n" + "=" * 50)
        logger.info("INGESTION COMPLETE")
        logger.info(f"  Chunks created: {len(chunks)}")
        logger.info(f"  Total vectors: {stats['total_vectors']}")
        logger.info(f"  Total documents: {stats['total_documents']}")
        logger.info(f"  Index path: {stats['index_path']}")
        logger.info("=" * 50)

    except Exception as e:
        import traceback
        logger.error(f"Ingestion failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()