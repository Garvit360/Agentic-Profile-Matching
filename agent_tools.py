import json
import logging
import re
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, ValidationError
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from domain_models import (
    AnalysisStatus,
    RequirementSet,
    format_excerpt_for_display,
    stable_candidate_id,
)
from eligibility import combined_score, recommendation_from_scores
from embedding_utils import require_openai_key
from job_matcher import run_job_matcher
from requirements_engine import RequirementSpecPayload, merge_initial_jd_requirements, spec_to_requirements

logger = logging.getLogger(__name__)

CHAT_MODEL: str = "gpt-4o-mini"


class RequirementSpec(BaseModel):
    must_have: list[str] = Field(default_factory=list, description="Must-have requirements")
    nice_to_have: list[str] = Field(default_factory=list, description="Nice-to-have requirements")
    years_experience: Optional[int] = Field(None, description="Minimum years of experience")
    role_context: str = Field(default="")


class CandidateAnalysis(BaseModel):
    summary: str = Field(..., description="Short summary of fit")
    strengths: list[str] = Field(default_factory=list, description="Key strengths")
    gaps: list[str] = Field(default_factory=list, description="Key gaps")
    fit_score: int = Field(..., ge=0, le=100, description="Fit score 0-100")
    analysis_status: AnalysisStatus = AnalysisStatus.COMPLETE


class CandidateShortlistItem(BaseModel):
    candidate_id: str = Field(..., description="Stable identifier")
    name: str = Field(..., description="Candidate name")
    resume_path: str = Field(..., description="Path to resume file")
    match_score: int = Field(..., ge=0, le=100, description="Initial match score")
    matched_skills: list[str] = Field(default_factory=list)
    relevant_excerpts: list[str] = Field(default_factory=list)
    reasoning: str = Field(..., description="Round 1 reasoning")
    round2_summary: str = Field(default="", description="Round 2 summary")
    strengths: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    fit_score: int = Field(default=0, ge=0, le=100)
    combined_score: int = Field(default=0, ge=0, le=100)
    recommendation: str = Field(default="pending", description="hire | no-hire | no-hire for now | pending")
    city: Optional[str] = Field(None, description="Best-effort inferred city")
    improvement_suggestions: list[str] = Field(default_factory=list)
    eligibility: str = Field(default="eligible")
    analysis_status: AnalysisStatus = AnalysisStatus.INCOMPLETE
    criterion_assessments: list[dict] = Field(default_factory=list)


class InterviewQuestions(BaseModel):
    questions: list[str] = Field(default_factory=list, description="Screening questions")


class SearchResultBundle(BaseModel):
    eligible: list[dict] = Field(default_factory=list)
    pending_review: list[dict] = Field(default_factory=list)


def _get_llm() -> ChatOpenAI:
    require_openai_key()
    return ChatOpenAI(model=CHAT_MODEL, temperature=0.2, timeout=120)


def infer_city_from_text(excerpts: list[str]) -> Optional[str]:
    patterns = [
        r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,\s*[A-Z]{2})\b",
        r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*,\s*(?:USA|United States)\b",
    ]
    for text in excerpts:
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
    return None


def _extract_json_text(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start = min([idx for idx in [text.find("{"), text.find("[")] if idx != -1], default=-1)
    if start > 0:
        text = text[start:]
    return text.strip()


def _load_json_payload(raw: str) -> dict:
    cleaned = _extract_json_text(raw)
    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise ValueError("Expected JSON object payload from model response.")
    return parsed


def extract_requirements(jd: str) -> RequirementSpec:
    llm = _get_llm()
    system = (
        "You extract structured requirements from job descriptions. "
        "Return ONLY JSON matching this schema: "
        '{"must_have": [string], "nice_to_have": [string], '
        '"years_experience": number|null, "role_context": string}.'
    )
    human = f"Job description:\n{jd}\n\nReturn JSON only."
    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)]).content
    payload = _load_json_payload(str(response))
    return RequirementSpec.model_validate(payload)


def requirements_from_spec(spec: RequirementSpec) -> RequirementSet:
    return spec_to_requirements(
        RequirementSpecPayload(
            must_have=spec.must_have,
            nice_to_have=spec.nice_to_have,
            years_experience=spec.years_experience,
            role_context=spec.role_context,
        )
    )


def requirements_from_jd(jd: str, spec: RequirementSpec) -> RequirementSet:
    return merge_initial_jd_requirements(
        jd,
        RequirementSpecPayload(
            must_have=spec.must_have,
            nice_to_have=spec.nice_to_have,
            years_experience=spec.years_experience,
            role_context=spec.role_context,
        ),
    )


def search_resumes(requirements: RequirementSet, jd: str, k: int = 10) -> SearchResultBundle:
    output = run_job_matcher(job_description=jd, k=k, requirements=requirements)
    return SearchResultBundle(
        eligible=output.get("top_matches", []),
        pending_review=output.get("pending_review", []),
    )


def analyze_candidate_fit(jd: str, candidate: dict, retry: bool = True) -> CandidateAnalysis:
    llm = _get_llm()
    excerpts = candidate.get("relevant_excerpts", [])
    evidence = "\n".join(
        f"- {format_excerpt_for_display(e, max_len=500)}" for e in excerpts[:8]
    )
    system = (
        "You are a hiring assistant. Analyze candidate fit for the job description using ONLY provided evidence. "
        "Return ONLY JSON: {\"summary\": string, \"strengths\": [string], "
        "\"gaps\": [string], \"fit_score\": number 0-100}."
    )
    human = (
        "Job description:\n"
        f"{jd}\n\n"
        "Candidate info:\n"
        f"Name: {candidate.get('candidate_name', 'Unknown')}\n"
        f"Matched skills: {candidate.get('matched_skills', [])}\n"
        f"Evidence excerpts:\n{evidence}\n"
        f"Round-1 reasoning: {candidate.get('reasoning', '')}\n\n"
        "Return JSON only."
    )

    def _invoke() -> CandidateAnalysis:
        response = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)]).content
        payload = _load_json_payload(str(response))
        parsed = CandidateAnalysis.model_validate(payload)
        return parsed.model_copy(update={"analysis_status": AnalysisStatus.COMPLETE})

    try:
        return _invoke()
    except (ValueError, json.JSONDecodeError, ValidationError) as first_err:
        logger.warning(
            "Analysis parse failed for candidate %s: %s",
            candidate.get("candidate_id", "unknown"),
            first_err,
        )
        if not retry:
            raise
        corrective = (
            f"{human}\n\nPrevious response was invalid JSON. Return valid JSON only with the required keys."
        )
        try:
            response = llm.invoke(
                [SystemMessage(content=system), HumanMessage(content=corrective)]
            ).content
            payload = _load_json_payload(str(response))
            parsed = CandidateAnalysis.model_validate(payload)
            return parsed.model_copy(update={"analysis_status": AnalysisStatus.COMPLETE})
        except (ValueError, json.JSONDecodeError, ValidationError) as second_err:
            logger.error(
                "Analysis incomplete for candidate %s after retry: %s",
                candidate.get("candidate_id", "unknown"),
                second_err,
            )
            return CandidateAnalysis(
                summary="Deep analysis could not be completed.",
                strengths=[],
                gaps=["Analysis incomplete — insufficient validated evidence"],
                fit_score=0,
                analysis_status=AnalysisStatus.INCOMPLETE,
            )


def generate_improvement_suggestions(jd: str, candidate_name: str, gaps: list[str]) -> list[str]:
    llm = _get_llm()
    system = (
        "You suggest concise, actionable improvement ideas for borderline candidates. "
        "Return ONLY JSON: {\"suggestions\": [string]}."
    )
    human = (
        "Job description:\n"
        f"{jd}\n\n"
        f"Candidate: {candidate_name}\n"
        f"Gaps: {gaps}\n\n"
        "Provide 2-4 suggestions. Return JSON only."
    )
    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)]).content
    payload = _load_json_payload(str(response))
    return [s for s in payload.get("suggestions", []) if isinstance(s, str)]


def compare_candidates_table(candidate_ids: list[str], shortlist: list[CandidateShortlistItem]) -> str:
    selected = [c for c in shortlist if c.candidate_id in set(candidate_ids)]
    if len(selected) < 2:
        return "Please provide at least two valid candidate IDs to compare."

    selected.sort(key=lambda c: c.combined_score, reverse=True)
    header = "| Rank | Name | Combined | Match | Fit | Recommendation | Key strengths |"
    sep = "|---:|---|---:|---:|---:|---|---|"
    rows: list[str] = [header, sep]
    for rank, c in enumerate(selected, start=1):
        strengths = ", ".join(c.strengths[:3]) or "N/A"
        rows.append(
            f"| {rank} | {c.name} | {c.combined_score} | {c.match_score} | {c.fit_score} | "
            f"{c.recommendation} | {strengths} |"
        )
    rows.append("")
    rows.append(
        "Table order follows stored combined scores (screening support — human judgment required)."
    )
    return "\n".join(rows)


def compare_candidates(candidate_ids: list[str], shortlist: list[CandidateShortlistItem]) -> str:
    return compare_candidates_table(candidate_ids, shortlist)


def generate_interview_questions_for_id(
    candidate_id: str,
    shortlist: list[CandidateShortlistItem],
    requirements: RequirementSet | None,
) -> list[str]:
    target = next((c for c in shortlist if c.candidate_id == candidate_id), None)
    if target is None:
        raise ValueError(f"Unknown candidate id: {candidate_id}")
    return generate_interview_questions(target, requirements)


def generate_interview_questions(
    candidate: CandidateShortlistItem,
    requirements: RequirementSet | None = None,
) -> list[str]:
    llm = _get_llm()
    role_bits = []
    if requirements:
        role_bits.extend(requirements.must_have_labels())
        if requirements.minimum_experience_years:
            role_bits.append(f"{requirements.minimum_experience_years}+ years")
    system = (
        "You generate interview screening questions tailored to a candidate and role. "
        "Return ONLY JSON: {\"questions\": [string]}."
    )
    human = (
        "Role requirements:\n"
        f"{', '.join(role_bits) or 'General role fit'}\n\n"
        "Candidate info:\n"
        f"Name: {candidate.name}\n"
        f"Strengths: {candidate.strengths}\n"
        f"Gaps: {candidate.gaps}\n"
        f"Summary: {candidate.round2_summary}\n"
        f"Evidence excerpts: {candidate.relevant_excerpts[:5]}\n\n"
        "Provide 5-7 concise questions grounded in the evidence. Return JSON only."
    )
    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)]).content
    payload = _load_json_payload(str(response))
    parsed = InterviewQuestions.model_validate(payload)
    questions = [q for q in parsed.questions if isinstance(q, str)]
    if len(questions) < 5:
        raise ValueError("Expected at least 5 interview questions from model.")
    return questions[:7]


def build_candidate_id(name: str, resume_path: str, index: int = 0) -> str:
    del name, index
    rel = resume_path.replace("\\", "/")
    return stable_candidate_id(rel)
