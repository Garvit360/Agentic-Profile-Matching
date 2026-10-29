"""Local cache for validated candidate profile extractions."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from domain_models import CandidateProfile

CACHE_DIR = Path(".profile_cache")


class CacheEntry(BaseModel):
    content_hash: str
    model: str
    profile: CandidateProfile


def cache_path_for(source_path: str) -> Path:
    safe = source_path.replace("/", "__")
    return CACHE_DIR / f"{safe}.json"


def load_cached_profile(
    source_path: str,
    content_hash: str,
    model: str,
) -> CandidateProfile | None:
    path = cache_path_for(source_path)
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    entry = CacheEntry.model_validate(raw)
    if entry.content_hash != content_hash:
        return None
    if entry.model != model:
        return None
    return entry.profile


def save_cached_profile(
    source_path: str,
    content_hash: str,
    model: str,
    profile: CandidateProfile,
) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    entry = CacheEntry(
        content_hash=content_hash,
        model=model,
        profile=profile,
    )
    cache_path_for(source_path).write_text(entry.model_dump_json(indent=2), encoding="utf-8")
