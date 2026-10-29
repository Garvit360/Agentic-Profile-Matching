"""Shared domain types for requirements, evidence, and screening."""

from __future__ import annotations

import hashlib
import re
from enum import Enum
from pathlib import Path
from typing import Optional, Literal

from pydantic import BaseModel, Field


class CriterionKind(str, Enum):
    SKILL = "skill"
    EXPERIENCE = "experience"
    OTHER = "other"


class Criterion(BaseModel):
    id: str = Field(..., description="Stable criterion identifier")
    label: str = Field(..., description="Human-readable requirement text")
    kind: CriterionKind = Field(default=CriterionKind.OTHER)


class RequirementSet(BaseModel):
    role_context: str = Field(default="", description="Role title or context")
    must_have: list[Criterion] = Field(default_factory=list)
    nice_to_have: list[Criterion] = Field(default_factory=list)
    minimum_experience_years: Optional[int] = Field(None, description="Minimum years required")

    def must_have_labels(self) -> list[str]:
        return [c.label for c in self.must_have]

    def nice_to_have_labels(self) -> list[str]:
        return [c.label for c in self.nice_to_have]


class AssessmentStatus(str, Enum):
    MET = "met"
    NOT_MET = "not_met"
    UNKNOWN = "unknown"


class CriterionAssessment(BaseModel):
    criterion_id: str
    status: AssessmentStatus
    evidence_excerpt: Optional[str] = None
    page_ref: Optional[int] = None


class ExtractionStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    PENDING = "pending"


class CandidateFact(BaseModel):
    kind: Literal["skill", "experience", "other"]
    claim: str
    excerpt: str
    page: int = Field(..., ge=1)


class CandidateProfile(BaseModel):
    candidate_id: str
    source_path: str = Field(..., description="Corpus-relative resume path")
    content_hash: str
    facts: list[CandidateFact] = Field(default_factory=list)
    extraction_status: ExtractionStatus = ExtractionStatus.PENDING
    failure_reason: Optional[str] = None


class EligibilityStatus(str, Enum):
    ELIGIBLE = "eligible"
    INELIGIBLE = "ineligible"
    PENDING_REVIEW = "pending_review"


class AnalysisStatus(str, Enum):
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"


class FinalRecommendation(str, Enum):
    HIRE = "hire"
    NO_HIRE = "no_hire"
    NO_HIRE_FOR_NOW = "no_hire_for_now"
    PENDING = "pending"


class RequestStatus(str, Enum):
    IDLE = "idle"
    IN_PROGRESS = "in_progress"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


# UTF-8 punctuation mis-decoded as Windows-1252 (common in PDF text extraction).
_MOJIBAKE_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("\u00e2\u20ac\u201c", "–"),
    ("\u00e2\u20ac\u201d", "—"),
    ("\u00e2\u20ac\u2122", "'"),
    ("\u00e2\u20ac\u0153", '"'),
    ("\u00e2\u20ac\u009d", '"'),
    ("\u00e2\u20ac\u00a2", "•"),
    ("\u00c2", ""),
)


def repair_common_mojibake(text: str) -> str:
    repaired = text
    for bad, good in _MOJIBAKE_REPLACEMENTS:
        repaired = repaired.replace(bad, good)
    return repaired


def format_excerpt_for_display(text: str, max_len: int = 280) -> str:
    """Single-line, terminal-friendly snippet from raw PDF/chunk text."""
    cleaned = repair_common_mojibake(normalize_whitespace(text))
    if max_len > 0 and len(cleaned) > max_len:
        return cleaned[: max_len - 1].rstrip() + "…"
    return cleaned


def excerpt_in_source(excerpt: str, source_text: str) -> bool:
    if not excerpt.strip():
        return False
    norm_excerpt = normalize_whitespace(excerpt).lower()
    norm_source = normalize_whitespace(source_text).lower()
    return norm_excerpt in norm_source


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def slugify(text: str) -> str:
    lowered = text.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "-", lowered).strip("-")
    return slug or "candidate"


def corpus_relative_path(resume_path: str, corpus_root: Path) -> str:
    path = Path(resume_path).resolve()
    root = corpus_root.resolve()
    try:
        rel = path.relative_to(root)
        return str(rel).replace("\\", "/")
    except ValueError:
        return Path(resume_path).name


def stable_candidate_id(source_path: str) -> str:
    """Derive ID from normalized corpus-relative path only."""
    normalized = source_path.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    normalized = normalized.lstrip("/")
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:10]
    stem = slugify(Path(normalized).stem)
    return f"{stem}-{digest}"


def criterion_id_from_label(label: str, kind: CriterionKind = CriterionKind.OTHER) -> str:
    return f"{kind.value}-{slugify(label)[:40]}"


def requirement_set_from_spec(
    must_have: list[str],
    nice_to_have: list[str],
    years_experience: Optional[int],
    role_context: str = "",
) -> RequirementSet:
    must_criteria = [
        Criterion(
            id=criterion_id_from_label(label, CriterionKind.SKILL if "year" not in label.lower() else CriterionKind.EXPERIENCE),
            label=label,
            kind=CriterionKind.EXPERIENCE if re.search(r"\d+\s*\+?\s*years?", label, re.I) else CriterionKind.SKILL,
        )
        for label in must_have
        if label.strip()
    ]
    nice_criteria = [
        Criterion(id=criterion_id_from_label(label, CriterionKind.SKILL), label=label, kind=CriterionKind.SKILL)
        for label in nice_to_have
        if label.strip()
    ]
    return RequirementSet(
        role_context=role_context,
        must_have=must_criteria,
        nice_to_have=nice_criteria,
        minimum_experience_years=years_experience,
    )
