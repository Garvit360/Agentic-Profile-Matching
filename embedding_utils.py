import logging
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT: Path = Path(__file__).resolve().parent

load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s: %(message)s",
)

CHROMA_PERSIST_DIR: str = str(PROJECT_ROOT / "chroma_db")
CHROMA_COLLECTION_NAME: str = "resume_chunks"


def require_openai_key() -> None:
    """Raise with a clear message if OPENAI_API_KEY is not set."""
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to a .env file in the project root or run:\n"
            "  export OPENAI_API_KEY=your-key-here"
        )
