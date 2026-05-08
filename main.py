"""
GenAI RAG System - Main Entry Point.
Production-level RAG application with LLM evaluation capabilities.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)


def initialize_system():
    """Initialize the RAG system and verify configuration."""
    settings = get_settings()

    logger.info("=" * 60)
    logger.info(f"  {settings.app_name} v{settings.app_version}")
    logger.info(f"  Environment: {settings.environment}")
    if settings.use_openrouter and settings.openrouter_api_key:
        logger.info(f"  LLM Provider: OpenRouter ({settings.openrouter_model})")
    elif settings.azure_openai_endpoint:
        logger.info(f"  LLM Provider: Azure OpenAI ({settings.azure_openai_deployment})")
    else:
        logger.info(f"  LLM Provider: OpenAI ({settings.openai_model})")
    logger.info(f"  Vector Store: {settings.vector_store_type}")
    logger.info("=" * 60)

    # Validate critical configuration
    has_valid_key = (
        (settings.use_openrouter and settings.openrouter_api_key)
        or settings.azure_openai_endpoint
        or (settings.openai_api_key and settings.openai_api_key != "your-openai-api-key-here")
    )
    if not has_valid_key:
        logger.warning("No valid LLM API key configured. Set OPENAI_API_KEY or OPENROUTER_API_KEY in .env file.")

    return settings


def main():
    """Main application entry point."""
    settings = initialize_system()

    # Example usage - query the RAG pipeline
    from pipeline.rag_pipeline import RAGPipeline
    from schemas.pipeline import QueryRequest

    try:
        pipeline = RAGPipeline()

        # Interactive query mode
        print("\n🤖 GenAI RAG System - Interactive Mode")
        print("Type 'quit' to exit, 'stats' for vector store info\n")

        while True:
            query = input("Question: ").strip()

            if query.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break
            elif query.lower() == "stats":
                stats = pipeline.vector_store.get_stats()
                print(f"  Vectors: {stats['total_vectors']}")
                print(f"  Documents: {stats['total_documents']}")
                continue
            elif not query:
                continue

            request = QueryRequest(query=query)
            response = pipeline.query(request)

            print(f"\nAnswer: {response.answer}")
            print(f"  (Sources: {len(response.sources)}, Time: {response.total_time_ms:.0f}ms)\n")

    except KeyboardInterrupt:
        print("\nShutdown requested.")
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()