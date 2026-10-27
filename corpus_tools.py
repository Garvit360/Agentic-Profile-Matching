"""Read-only filesystem access to the configured resume corpus."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field

from domain_models import corpus_relative_path


class CorpusFileInfo(BaseModel):
    relative_path: str
    name: str
    size_bytes: int


class CorpusPage(BaseModel):
    page_number: int = Field(..., ge=1)
    text: str


class CorpusDocument(BaseModel):
    relative_path: str
    pages: list[CorpusPage]


def resolve_corpus_root(configured: str | Path | None = None) -> Path:
    from embedding_utils import PROJECT_ROOT

    if configured is not None:
        root = Path(configured).resolve()
    elif os.environ.get("RESUME_CORPUS_DIR"):
        root = Path(os.environ["RESUME_CORPUS_DIR"]).resolve()
    else:
        root = (PROJECT_ROOT / "Resume").resolve()
    if not root.exists():
        raise FileNotFoundError(f"Resume corpus not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Resume corpus is not a directory: {root}")
    return root


def _is_safe_path(path: Path, corpus_root: Path) -> bool:
    resolved = path.resolve()
    root = corpus_root.resolve()
    if not str(resolved).startswith(str(root)):
        return False
    if resolved.is_symlink():
        target = resolved.resolve()
        if not str(target).startswith(str(root)):
            return False
    return True


def list_corpus_files(corpus_root: Path | None = None) -> list[CorpusFileInfo]:
    root = corpus_root or resolve_corpus_root()
    files: list[CorpusFileInfo] = []
    for pdf in sorted(root.rglob("*.pdf")):
        if not _is_safe_path(pdf, root):
            continue
        rel = corpus_relative_path(str(pdf), root)
        files.append(
            CorpusFileInfo(
                relative_path=rel,
                name=pdf.name,
                size_bytes=pdf.stat().st_size,
            )
        )
    return files


def search_corpus_filenames(query: str, corpus_root: Path | None = None) -> list[str]:
    root = corpus_root or resolve_corpus_root()
    q = query.lower().strip()
    if not q:
        return []
    matches: list[str] = []
    for info in list_corpus_files(root):
        if q in info.name.lower() or q in info.relative_path.lower():
            matches.append(info.relative_path)
    return matches


def read_resume_pdf(relative_path: str, corpus_root: Path | None = None) -> CorpusDocument:
    from langchain_community.document_loaders import PyPDFLoader

    root = corpus_root or resolve_corpus_root()
    full_path = (root / relative_path).resolve()
    if not _is_safe_path(full_path, root):
        raise PermissionError(f"Path outside corpus: {relative_path}")
    if not full_path.exists():
        raise FileNotFoundError(f"Resume not found: {relative_path}")
    loader = PyPDFLoader(str(full_path))
    docs = loader.load()
    pages = [
        CorpusPage(page_number=idx + 1, text=doc.page_content or "")
        for idx, doc in enumerate(docs)
    ]
    return CorpusDocument(relative_path=relative_path, pages=pages)
