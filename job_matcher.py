"""
Part B: Job Matching Engine (LangChain).
Semantic retrieval via LangChain Chroma, eligibility-aware ranking.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pydantic import BaseModel, Field
from langchain_openai import OpenAIEmbeddings
import chromadb

from domain_models import RequirementSet, stable_candidate_id, CandidateProfile
from eligibility import assess_eligibility
from embedding_utils import CHROMA_COLLECTION_NAME, CHROMA_PERSIST_DIR, require_openai_key

logger = logging.getLogger(__name__)

INITIAL_WEIGHT = 0.6
DEEP_WEIGHT = 0.4
HIRE_THRESHOLD = 80
BORDERLINE_THRESHOLD = 60


class JobMatchResult(BaseModel):
    candidate_name: str = Field(..., description="Candidate full name")
    resume_path: str = Field(..., description="Path to resume file")
    candidate_id: str = Field(..., description="Stable candidate identifier")
    match_score: int = Field(..., ge=0, le=100, description="Match score 0-100")
    matched_skills: list[str] = Field(default_factory=list, description="Skills that matched JD")
    relevant_excerpts: list[str] = Field(default_factory=list, description="Relevant text snippets")
    reasoning: str = Field(..., description="Why this candidate matches")
    eligibility: str = Field(default="eligible")
    criterion_assessments: list[dict] = Field(default_factory=list)


class JobMatchOutput(BaseModel):
    job_description: str = Field(..., description="Input job description")
    top_matches: list[JobMatchResult] = Field(default_factory=list, description="Top K eligible matches")
    pending_review: list[JobMatchResult] = Field(default_factory=list, description="Pending review matches")


class IndexRebuildRequired(ValueError):
    """The on-disk corpus does not contain the required screening evidence."""


def validate_index_metadata(metadata: dict) -> None:
    if "profile" not in metadata:
        raise IndexRebuildRequired(
            "This resume index predates structured candidate evidence. "
            "Rebuild it with: python resume_rag.py Resume. "
            "The rebuild requires OpenAI access; no profiles will be fabricated."
        )
    CandidateProfile.model_validate_json(metadata["profile"])


def _semantic_search_all_chunks(
    query: str,
    persist_dir: str,
    fetch_k: int,
) -> list[dict]:
    persist_path = Path(persist_dir)
    if not persist_path.exists() or not list(persist_path.iterdir()):
        logger.error(
            "Vector database not found at %s\n"
            "  → Run step 1 first: python resume_rag.py\n"
            "  → This indexes PDFs from the Resume/ folder and creates the database.",
            persist_dir,
        )
        raise FileNotFoundError(
            f"Chroma DB not found at {persist_dir}. Run resume_rag.py first to index resumes."
        )

    embedding = OpenAIEmbeddings(model="text-embedding-3-small")
    collection = chromadb.PersistentClient(path=persist_dir).get_collection(CHROMA_COLLECTION_NAME)
    for metadata in collection.get(include=["metadatas"])["metadatas"]:
        validate_index_metadata(metadata)
    count = collection.count()
    if count == 0:
        raise ValueError("Resume index is empty. Run python resume_rag.py Resume.")
    hits = collection.query(query_embeddings=[embedding.embed_query(query)], n_results=count,
                            include=["documents", "metadatas", "distances"])
    return [{"document": document, "metadata": metadata, "distance": distance}
            for document, metadata, distance in zip(hits["documents"][0], hits["metadatas"][0], hits["distances"][0])]


def _requirements_query(requirements: RequirementSet, job_description: str) -> str:
    parts = [requirements.role_context, job_description]
    parts.extend(requirements.must_have_labels())
    parts.extend(requirements.nice_to_have_labels())
    if requirements.minimum_experience_years is not None:
        parts.append(f"{requirements.minimum_experience_years}+ years experience")
    return "\n".join(p for p in parts if p.strip())


def _matched_skills_from_meta(meta: dict, requirements: RequirementSet) -> list[str]:
    meta_skills_str = meta.get("skills") or ""
    meta_skills_list = [x.strip() for x in meta_skills_str.split(",") if x.strip()]
    labels = [l.lower() for l in requirements.must_have_labels() + requirements.nice_to_have_labels()]
    matched: list[str] = []
    for sk in meta_skills_list:
        sk_l = sk.lower()
        if any(lab in sk_l or sk_l in lab for lab in labels):
            matched.append(sk)
    return matched


def score_match(
    requirements: RequirementSet,
    resume_metadata: dict,
    chunks: list[dict],
    base_distance: float,
) -> tuple[int, str, list[str], list[str]]:
    similarity_score = max(0.0, 100.0 - base_distance * 50.0)
    matched_skills = _matched_skills_from_meta(resume_metadata, requirements)
    boost = min(20, len(matched_skills) * 3)
    for crit in requirements.nice_to_have:
        if any(crit.label.lower() in sk.lower() for sk in matched_skills):
            boost += 2
    final_score = min(100, int(similarity_score) + boost)
    from domain_models import format_excerpt_for_display

    relevant_excerpts = [
        format_excerpt_for_display(c.get("document", c.get("text", "")), max_len=400)
        for c in chunks[:5]
    ]
    relevant_excerpts = [e for e in relevant_excerpts if e]
    reasoning_parts: list[str] = []
    if matched_skills:
        reasoning_parts.append(f"Matched skills: {', '.join(matched_skills[:10])}.")
    exp = resume_metadata.get("experience_years")
    if exp is not None and exp >= 0:
        reasoning_parts.append(f"Experience: {exp} years.")
    edu = resume_metadata.get("education")
    if edu:
        reasoning_parts.append(f"Education: {edu[:80]}.")
    reasoning = " ".join(reasoning_parts) if reasoning_parts else "Semantic match based on resume content."
    return final_score, reasoning, matched_skills, relevant_excerpts


def run_job_matcher(
    job_description: str,
    k: int = 10,
    persist_dir: str = CHROMA_PERSIST_DIR,
    requirements: RequirementSet | None = None,
    fetch_multiplier: int = 50,
) -> dict:
    require_openai_key()
    from requirements_engine import RequirementSpecPayload, merge_initial_jd_requirements

    if requirements is None:
        from agent_tools import extract_requirements, requirements_from_jd
        requirements = requirements_from_jd(job_description, extract_requirements(job_description))

    query = _requirements_query(requirements, job_description)
    fetch_k = max(k * fetch_multiplier, 200)
    logger.info("Searching for matches (fetch up to %d chunks, top %d eligible) ...", fetch_k, k)
    raw_results = _semantic_search_all_chunks(query, persist_dir=persist_dir, fetch_k=fetch_k)

    by_path: dict[str, list[dict]] = {}
    for r in raw_results:
        path = (r.get("metadata") or {}).get("path") or "unknown"
        by_path.setdefault(path, []).append(r)

    scored_eligible: list[JobMatchResult] = []
    pending_review: list[JobMatchResult] = []

    for path, chunk_list in by_path.items():
        meta = (chunk_list[0].get("metadata")) or {}
        rel_path = path
        cid = stable_candidate_id(rel_path.replace("\\", "/"))
        validate_index_metadata(meta)
        profile = CandidateProfile.model_validate_json(meta["profile"])
        skill_facts = [f for f in profile.facts if f.kind == "skill"]
        experience_facts = [f for f in profile.facts if f.kind == "experience"]
        years = {f.years for f in experience_facts}
        meta = {**meta, "skills": ",".join(f.claim for f in skill_facts),
                "experience_years": next(iter(years)) if len(years) == 1 else None}
        eligibility, assessments = assess_eligibility(meta, requirements)
        criteria = {c.id: c.label.casefold() for c in requirements.must_have}
        for assessment in assessments:
            if assessment.criterion_id == "experience-minimum":
                evidence = experience_facts if len(years) == 1 else []
            else:
                evidence = [f for f in skill_facts if f.claim.casefold() == criteria[assessment.criterion_id]]
            if evidence:
                assessment.evidence_excerpt = evidence[0].excerpt
                assessment.page_ref = evidence[0].page
        best_dist = min(c.get("distance", 2.0) for c in chunk_list)
        score, reasoning, matched_skills, excerpts = score_match(requirements, meta, chunk_list, best_dist)
        result = JobMatchResult(
            candidate_name=meta.get("name") or Path(path).stem.replace("_", " "),
            resume_path=path,
            candidate_id=cid,
            match_score=min(100, score),
            matched_skills=matched_skills,
            relevant_excerpts=excerpts,
            reasoning=reasoning,
            eligibility=eligibility.value,
            criterion_assessments=[a.model_dump() for a in assessments],
        )
        if eligibility.value == "eligible":
            scored_eligible.append(result)
        elif eligibility.value == "pending_review":
            pending_review.append(result)

    scored_eligible.sort(key=lambda x: -x.match_score)
    pending_review.sort(key=lambda x: -x.match_score)
    top = scored_eligible[:k]
    pending_top = pending_review[:10]

    logger.info(
        "Found %d eligible and %d pending-review candidate(s) after scoring %d profiles.",
        len(top),
        len(pending_top),
        len(by_path),
    )
    output = JobMatchOutput(
        job_description=job_description,
        top_matches=top,
        pending_review=pending_top,
    )
    return output.model_dump()


if __name__ == "__main__":
    import json
    import sys

    jd = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "Senior Python developer. 5+ years Python. Machine Learning experience required."
    )
    try:
        result = run_job_matcher(jd, k=10)
        print(json.dumps(result, indent=2))
    except FileNotFoundError as e:
        logger.error("%s", e)
        sys.exit(1)
    except RuntimeError as e:
        logger.error("%s", e)
        sys.exit(1)
