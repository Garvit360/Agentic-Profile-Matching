import logging
import os
import re
import sys
import warnings
from typing import Optional

warnings.filterwarnings(
    "ignore",
    message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater.",
    category=UserWarning,
)

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_core._api.deprecation import LangChainDeprecationWarning

from agent_tools import (
    CandidateShortlistItem,
    RequirementSpec,
    analyze_candidate_fit,
    build_candidate_id,
    compare_candidates,
    extract_requirements,
    generate_improvement_suggestions,
    generate_interview_questions,
    requirements_from_jd,
    search_resumes,
)
from domain_models import (
    AnalysisStatus,
    EligibilityStatus,
    RequestStatus,
    RequirementSet,
    format_excerpt_for_display,
)
from eligibility import combined_score, recommendation_from_scores
from requirements_engine import apply_refinement


class Message(BaseModel):
    role: str = Field(..., description="user | assistant")
    content: str = Field(..., description="message content")


class ResultSnapshot(BaseModel):
    requirements: Optional[RequirementSet] = None
    shortlist: list[CandidateShortlistItem] = Field(default_factory=list)
    pending_review: list[CandidateShortlistItem] = Field(default_factory=list)


class AgentState(BaseModel):
    conversation_history: list[Message] = Field(default_factory=list)
    job_description: Optional[str] = None
    requirements: Optional[RequirementSet] = None
    requirements_spec: Optional[RequirementSpec] = None
    candidate_pool: list[dict] = Field(default_factory=list)
    pending_pool: list[dict] = Field(default_factory=list)
    shortlist: list[CandidateShortlistItem] = Field(default_factory=list)
    pending_review: list[CandidateShortlistItem] = Field(default_factory=list)
    last_successful_snapshot: Optional[ResultSnapshot] = None
    previous_snapshot: Optional[ResultSnapshot] = None
    request_status: RequestStatus = RequestStatus.IDLE
    last_user_intent: Optional[str] = None
    last_user_message: Optional[str] = None
    last_response: Optional[str] = None


# =============================================================================
# INTENT PARSING
# =============================================================================

def infer_intent(user_msg: str, has_jd: bool) -> str:
    msg = user_msg.lower().strip()

    if re.search(r"\b(list|show)\b.*\b(corpus|resumes|files)\b", msg) or msg.startswith("list corpus"):
        return "list_corpus"
    if re.search(r"\b(search|find)\b.*\b(file|filename|resume name)\b", msg):
        return "search_files"
    if re.search(r"\bread resume\b", msg):
        return "read_resume"
    if re.search(r"\bprevious (results|shortlist|snapshot)\b", msg):
        return "previous_results"

    if re.search(r"\bcompare\b", msg) or " vs " in msg or " versus " in msg:
        return "compare"
    if re.search(r"\binterview\b", msg) or (
        re.search(r"\bquestions\b", msg) and re.search(r"\b(candidate|top)\b", msg)
    ):
        return "questions"
    if msg.startswith("why") or "rank higher" in msg or "why did" in msg:
        return "why"
    if re.search(r"\b(details|show me the top \d+|top \d+ candidates)\b", msg) and not re.search(
        r"\b(find|search|match|looking for|candidates with)\b", msg
    ):
        return "details"

    if any(k in msg for k in ["actually", "refine", "update", "change", "require ", "remove ", "add ", " is optional"]):
        return "refine" if has_jd else "search"
    if any(k in msg for k in ["find", "search", "candidates", "match", "looking for"]):
        return "search"

    if has_jd:
        return "unsupported"
    return "search"


def parse_compare_targets(user_msg: str, shortlist: list[CandidateShortlistItem]) -> list[str]:
    msg = user_msg.lower()
    top_match = re.search(r"top\s+(\d+)", msg)
    if top_match:
        count = int(top_match.group(1))
        return [c.candidate_id for c in shortlist[:count]]

    if " vs " in msg:
        parts = msg.split(" vs ")
    elif " versus " in msg:
        parts = msg.split(" versus ")
    elif "compare" in msg and " and " in msg:
        parts = msg.split("compare", 1)[1].split(" and ")
    else:
        parts = []

    targets: list[str] = []
    for part in parts:
        name = part.strip().strip(".?")
        if not name:
            continue
        resolved = resolve_candidate_query(name, shortlist)
        if isinstance(resolved, CandidateShortlistItem):
            targets.append(resolved.candidate_id)
    return targets


def resolve_candidate_query(
    query: str,
    shortlist: list[CandidateShortlistItem],
) -> CandidateShortlistItem | str | None:
    q = query.lower().strip()
    if q in {"top", "top candidate", "the top candidate", "first candidate"}:
        return shortlist[0] if shortlist else None

    top_n = re.match(r"top\s+(\d+)$", q)
    if top_n:
        idx = int(top_n.group(1)) - 1
        if 0 <= idx < len(shortlist):
            return shortlist[idx]
        return f"No candidate at rank {idx + 1}."

    exact = [c for c in shortlist if c.candidate_id.lower() == q]
    if len(exact) == 1:
        return exact[0]

    name_matches = [c for c in shortlist if c.name.lower() == q]
    if len(name_matches) == 1:
        return name_matches[0]

    tokens = [t for t in re.split(r"\s+", q) if t]
    partial = [c for c in shortlist if all(t in c.name.lower() for t in tokens)]
    if len(partial) == 1:
        return partial[0]
    if len(partial) > 1:
        names = ", ".join(c.name for c in partial[:5])
        return f"Multiple candidates match '{query}': {names}. Please specify an id or full name."
    return None


def parse_why_targets(
    user_msg: str,
    shortlist: list[CandidateShortlistItem],
) -> tuple[Optional[CandidateShortlistItem], Optional[CandidateShortlistItem], Optional[str]]:
    msg = user_msg.strip()
    m = re.search(r"why did (.+?) rank higher than (.+?)\??$", msg, re.I)
    if not m:
        return None, None, "Use: Why did <name A> rank higher than <name B>?"

    first_raw, second_raw = m.group(1).strip(), m.group(2).strip()
    first = resolve_candidate_query(first_raw, shortlist)
    second = resolve_candidate_query(second_raw, shortlist)

    if isinstance(first, str):
        return None, None, first
    if isinstance(second, str):
        return None, None, second
    if first is None or second is None:
        missing = first_raw if first is None else second_raw
        return None, None, f"I could not find a unique candidate matching '{missing}'."
    return first, second, None


def parse_question_target(
    user_msg: str,
    shortlist: list[CandidateShortlistItem],
) -> CandidateShortlistItem | str | None:
    msg = user_msg.lower()
    if "top candidate" in msg or re.search(r"\bfor the top candidate\b", msg):
        return shortlist[0] if shortlist else None
    top_match = re.search(r"for\s+(?:the\s+)?top\s+(\d+)", msg)
    if top_match:
        idx = int(top_match.group(1)) - 1
        if 0 <= idx < len(shortlist):
            return shortlist[idx]
        return f"No candidate at rank {idx + 1}."

    for c in shortlist:
        if c.candidate_id.lower() in msg:
            return c

    name_candidates = [c for c in shortlist if c.name.lower() in msg]
    if len(name_candidates) == 1:
        return name_candidates[0]

    tokens = re.findall(r"[a-z]+", msg)
    for c in shortlist:
        name_tokens = c.name.lower().split()
        if name_tokens and all(t in tokens for t in name_tokens):
            return c

    quoted = re.search(r"for\s+([A-Za-z][A-Za-z\s'-]+)", user_msg)
    if quoted:
        resolved = resolve_candidate_query(quoted.group(1).strip(), shortlist)
        if isinstance(resolved, str):
            return resolved
        if resolved is not None:
            return resolved

    return None


def parse_top_count(user_msg: str, default: int = 3) -> int:
    match = re.search(r"top\s+(\d+)", user_msg.lower())
    if not match:
        return default
    return max(1, min(int(match.group(1)), 10))


def wants_previous_shortlist(user_msg: str) -> bool:
    msg = user_msg.lower()
    return bool(re.search(r"\bprevious\b|\bprior\b|\blast (search|results|shortlist)\b", msg))


def resolve_followup_shortlist(
    st: AgentState,
    user_msg: str,
) -> tuple[list[CandidateShortlistItem], Optional[str]]:
    if st.shortlist:
        return st.shortlist, None
    if wants_previous_shortlist(user_msg) and st.previous_snapshot and st.previous_snapshot.shortlist:
        return st.previous_snapshot.shortlist, (
            "Using the previous eligible shortlist (before the latest requirement change):\n\n"
        )
    return [], None


def no_shortlist_message(st: AgentState, *, action: str) -> str:
    if not st.job_description and not st.requirements:
        return (
            f"No candidates to {action} yet. Run a search first "
            "(e.g. 'Find candidates with React and 3+ years experience')."
        )

    pending = len(st.pending_review)
    prev_eligible = len(st.previous_snapshot.shortlist) if st.previous_snapshot else 0

    parts = [
        f"No eligible candidates to {action} under the current requirements (eligible ranked: 0)."
    ]
    if pending:
        parts.append(
            f"{pending} candidate(s) are in pending review with unverified must-haves; "
            f"{action} uses only the eligible ranked list, not pending review."
        )
    if prev_eligible:
        parts.append(
            f"Your prior search had {prev_eligible} eligible candidate(s). "
            "Try 'compare previous top 3', or relax requirements and search again."
        )
    elif not pending:
        parts.append("Try broadening must-haves or minimum experience, then search again.")
    return " ".join(parts)


def format_requirements(req: RequirementSet) -> list[str]:
    lines = ["Effective requirements:"]
    if req.role_context:
        lines.append(f"- Role: {req.role_context}")
    if req.minimum_experience_years is not None:
        lines.append(f"- Minimum experience: {req.minimum_experience_years}+ years")
    if req.must_have:
        lines.append("- Must-have:")
        lines.extend([f"  • {c.label}" for c in req.must_have])
    if req.nice_to_have:
        lines.append("- Nice-to-have:")
        lines.extend([f"  • {c.label}" for c in req.nice_to_have])
    return lines


def explain_snapshot_delta(previous: ResultSnapshot, current: ResultSnapshot) -> list[str]:
    if not previous.shortlist:
        return []

    prev_by_id = {c.candidate_id: c for c in previous.shortlist}
    curr_by_id = {c.candidate_id: c for c in current.shortlist}
    lines = ["Ranking changes (requirement-driven where noted):"]

    added = [cid for cid in curr_by_id if cid not in prev_by_id]
    removed = [cid for cid in prev_by_id if cid not in curr_by_id]
    if added:
        lines.append(f"- Added to eligible shortlist: {', '.join(curr_by_id[c].name for c in added)}")
    if removed:
        lines.append(f"- Removed from eligible shortlist: {', '.join(prev_by_id[c].name for c in removed)}")

    for cid, curr in curr_by_id.items():
        prev = prev_by_id.get(cid)
        if not prev:
            continue
        prev_rank = sorted(previous.shortlist, key=lambda c: c.combined_score, reverse=True)
        curr_rank = sorted(current.shortlist, key=lambda c: c.combined_score, reverse=True)
        pr = next(i for i, c in enumerate(prev_rank, 1) if c.candidate_id == cid)
        cr = next(i for i, c in enumerate(curr_rank, 1) if c.candidate_id == cid)
        if pr != cr or prev.combined_score != curr.combined_score:
            delta = curr.combined_score - prev.combined_score
            lines.append(
                f"- {curr.name}: rank {pr}→{cr}, combined score {prev.combined_score}→{curr.combined_score} "
                f"(Δ{delta:+d}). Score shifts may reflect updated requirements or deeper analysis."
            )
    return lines


def _shortlist_item_from_pool(
    candidate: dict,
    jd: str,
    requirements: RequirementSet,
) -> CandidateShortlistItem:
    jd = "\n".join(format_requirements(requirements))
    analysis = analyze_candidate_fit(jd, candidate)
    match_score = int(candidate.get("match_score", 0))
    combined = combined_score(match_score, analysis.fit_score)
    eligibility = EligibilityStatus(candidate.get("eligibility", "eligible"))
    complete = analysis.analysis_status == AnalysisStatus.COMPLETE
    recommendation = recommendation_from_scores(eligibility, complete, combined)

    suggestions: list[str] = []
    if recommendation == "no_hire_for_now" and complete:
        suggestions = generate_improvement_suggestions(
            jd,
            candidate.get("candidate_name", "Unknown"),
            analysis.gaps,
        )

    return CandidateShortlistItem(
        candidate_id=candidate.get("candidate_id")
        or build_candidate_id(
            candidate.get("candidate_name", "Unknown"),
            candidate.get("resume_path", ""),
        ),
        name=candidate.get("candidate_name", "Unknown"),
        resume_path=candidate.get("resume_path", ""),
        match_score=match_score,
        matched_skills=candidate.get("matched_skills", []),
        relevant_excerpts=candidate.get("relevant_excerpts", []),
        reasoning=candidate.get("reasoning", ""),
        round2_summary=analysis.summary,
        strengths=analysis.strengths,
        gaps=analysis.gaps,
        fit_score=analysis.fit_score,
        combined_score=combined,
        recommendation=recommendation,
        city=infer_city_from_text(candidate.get("relevant_excerpts", [])),
        improvement_suggestions=suggestions,
        eligibility=eligibility.value,
        analysis_status=analysis.analysis_status,
        criterion_assessments=candidate.get("criterion_assessments", []),
    )


def infer_city_from_text(excerpts: list[str]) -> Optional[str]:
    from agent_tools import infer_city_from_text as _infer

    return _infer(excerpts)


def _pending_item_from_pool(candidate: dict) -> CandidateShortlistItem:
    return CandidateShortlistItem(
        candidate_id=candidate.get("candidate_id")
        or build_candidate_id(
            candidate.get("candidate_name", "Unknown"),
            candidate.get("resume_path", ""),
        ),
        name=candidate.get("candidate_name", "Unknown"),
        resume_path=candidate.get("resume_path", ""),
        match_score=int(candidate.get("match_score", 0)),
        matched_skills=candidate.get("matched_skills", []),
        relevant_excerpts=candidate.get("relevant_excerpts", []),
        reasoning=candidate.get("reasoning", ""),
        recommendation="pending",
        eligibility="pending_review",
        analysis_status=AnalysisStatus.INCOMPLETE,
        criterion_assessments=candidate.get("criterion_assessments", []),
    )


# =============================================================================
# GRAPH NODES
# =============================================================================

def parse_input_node(state: dict) -> dict:
    st = AgentState.model_validate(state)
    user_msg = (st.last_user_message or "").strip()
    intent = infer_intent(user_msg, has_jd=bool(st.job_description))

    history = list(st.conversation_history)
    if user_msg:
        history.append(Message(role="user", content=user_msg))

    updates: dict = {
        "conversation_history": history,
        "last_user_intent": intent,
        "request_status": RequestStatus.IN_PROGRESS,
    }

    if intent == "search":
        updates["job_description"] = user_msg
        updates["candidate_pool"] = []
        updates["pending_pool"] = []
        updates["shortlist"] = []
        updates["pending_review"] = []

    return updates


def extract_requirements_node(state: dict) -> dict:
    st = AgentState.model_validate(state)
    if not st.job_description:
        return {
            "last_response": "Please provide a job description to extract requirements.",
            "request_status": RequestStatus.FAILED,
        }

    intent = st.last_user_intent or ""
    spec = extract_requirements(st.job_description)

    if intent == "refine" and st.requirements and (st.last_user_message or "").strip():
        refinement = apply_refinement(st.requirements, st.last_user_message or "")
        if refinement.clarification:
            return {
                "last_response": refinement.clarification,
                "request_status": RequestStatus.FAILED,
            }
        if refinement.applied:
            requirements = refinement.requirements
        else:
            requirements = requirements_from_jd(st.job_description, spec)
    else:
        requirements = requirements_from_jd(st.job_description, spec)

    return {"requirements": requirements, "requirements_spec": spec,
            "candidate_pool": [], "pending_pool": [], "shortlist": [], "pending_review": []}


def search_resumes_node(state: dict) -> dict:
    st = AgentState.model_validate(state)
    if not st.job_description or not st.requirements:
        return {
            "last_response": "Please provide a job description to search resumes.",
            "request_status": RequestStatus.FAILED,
        }
    bundle = search_resumes(st.requirements, st.job_description, k=10)
    return {
        "candidate_pool": bundle.eligible,
        "pending_pool": bundle.pending_review,
    }


def rank_candidates_node(state: dict) -> dict:
    st = AgentState.model_validate(state)
    if not st.job_description or not st.requirements:
        return {
            "last_response": "No requirements available to rank candidates.",
            "request_status": RequestStatus.FAILED,
            "shortlist": [],
            "pending_review": [],
        }

    if not st.candidate_pool and not st.pending_pool:
        return {
            "last_response": "No candidates matched the current requirements.",
            "request_status": RequestStatus.SUCCEEDED,
            "shortlist": [],
            "pending_review": [],
        }

    shortlist: list[CandidateShortlistItem] = []
    for candidate in st.candidate_pool[:10]:
        shortlist.append(_shortlist_item_from_pool(candidate, st.job_description, st.requirements))

    shortlist.sort(key=lambda c: c.combined_score, reverse=True)

    pending: list[CandidateShortlistItem] = []
    for candidate in st.pending_pool[:10]:
        pending.append(_pending_item_from_pool(candidate))

    return {"shortlist": shortlist, "pending_review": pending}


def generate_report_node(state: dict) -> dict:
    st = AgentState.model_validate(state)
    if st.request_status == RequestStatus.FAILED:
        return {}

    if not st.shortlist and not st.pending_review:
        return {
            "last_response": "No candidates matched the current requirements.",
            "request_status": RequestStatus.SUCCEEDED,
            "previous_snapshot": st.last_successful_snapshot,
            "last_successful_snapshot": ResultSnapshot(
                requirements=st.requirements,
                shortlist=[],
                pending_review=[],
            ),
        }

    lines = ["Screening recommendations (human judgment required):"]
    lines.append(f"- Eligible ranked: {len(st.shortlist)} | Pending review: {len(st.pending_review)}")

    if st.requirements:
        lines.extend(format_requirements(st.requirements))
        lines.append("")

    lines.append("Top eligible candidates (ranked):")
    for idx, c in enumerate(st.shortlist[:10], start=1):
        lines.append(
            f"{idx}. {c.name} (id: {c.candidate_id}) | combined={c.combined_score} | "
            f"match={c.match_score} | fit={c.fit_score} | {c.recommendation} | analysis={c.analysis_status.value}"
        )
        if c.strengths:
            lines.append(f"   Strengths: {', '.join(c.strengths[:4])}")
        if c.gaps:
            lines.append(f"   Gaps: {', '.join(c.gaps[:4])}")
        if c.recommendation == "no_hire_for_now" and c.improvement_suggestions:
            lines.append(f"   Suggestions: {'; '.join(c.improvement_suggestions[:3])}")

    if st.pending_review:
        lines.append("")
        lines.append("Pending review (unverified must-haves — not in ranked shortlist):")
        for c in st.pending_review[:10]:
            unknown = [
                a.get("criterion_id", "?")
                for a in c.criterion_assessments
                if a.get("status") == "unknown"
            ]
            lines.append(
                f"- {c.name} (id: {c.candidate_id}) | match={c.match_score} | needs evidence: "
                f"{', '.join(unknown[:4]) or 'experience/skills verification'}"
            )

    snapshot = ResultSnapshot(
        requirements=st.requirements,
        shortlist=st.shortlist,
        pending_review=st.pending_review,
    )
    if st.last_successful_snapshot:
        delta_lines = explain_snapshot_delta(st.last_successful_snapshot, snapshot)
        if len(delta_lines) > 1:
            lines.append("")
            lines.extend(delta_lines)

    lines.append("")
    lines.append(
        "Try: 'compare top 3', 'why did <A> rank higher than <B>', "
        "'show details for top 1', 'interview questions for top candidate'."
    )

    return {
        "last_response": "\n".join(lines),
        "request_status": RequestStatus.SUCCEEDED,
        "previous_snapshot": st.last_successful_snapshot,
        "last_successful_snapshot": snapshot,
    }


def compare_candidates_node(state: dict) -> dict:
    st = AgentState.model_validate(state)
    user_msg = st.last_user_message or ""
    shortlist, prefix = resolve_followup_shortlist(st, user_msg)
    if not shortlist:
        return {"last_response": no_shortlist_message(st, action="compare")}
    ids = parse_compare_targets(user_msg, shortlist)
    if len(ids) < 2:
        return {
            "last_response": "Please specify at least two candidates, e.g. 'compare top 3'.",
        }
    comparison = compare_candidates(ids, shortlist)
    body = f"{prefix}{comparison}" if prefix else comparison
    return {"last_response": body}


def explain_ranking_node(state: dict) -> dict:
    st = AgentState.model_validate(state)
    user_msg = st.last_user_message or ""
    shortlist, prefix = resolve_followup_shortlist(st, user_msg)
    if len(shortlist) < 1:
        return {"last_response": no_shortlist_message(st, action="explain ranking")}

    higher, lower, err = parse_why_targets(user_msg, shortlist)
    if err:
        return {"last_response": err}
    assert higher is not None and lower is not None

    if higher.candidate_id == lower.candidate_id:
        return {"last_response": "Please name two different candidates."}

    ordered = sorted([higher, lower], key=lambda c: c.combined_score, reverse=True)
    top, bottom = ordered[0], ordered[1]

    if top.candidate_id == bottom.candidate_id:
        return {"last_response": "Both candidates have the same combined score — ranking is tied."}

    if higher.combined_score == lower.combined_score:
        lines = [
            f"{higher.name} and {lower.name} are tied with combined score {higher.combined_score}.",
            "Either ordering is acceptable; compare strengths and gaps below.",
        ]
    elif higher.combined_score > lower.combined_score:
        lines = [f"{higher.name} ranks above {lower.name} in the current shortlist:"]
    else:
        lines = [
            f"Note: {lower.name} actually ranks above {higher.name} "
            f"({lower.combined_score} vs {higher.combined_score}). Explanation:"
        ]

    lines.extend(
        [
            f"- Combined scores: {higher.name}={higher.combined_score}, {lower.name}={lower.combined_score}.",
            f"- Strengths ({higher.name}): {', '.join(higher.strengths[:4]) or 'N/A'}.",
            f"- Gaps ({lower.name}): {', '.join(lower.gaps[:4]) or 'N/A'}.",
        ]
    )
    lines.append(f"- Evidence excerpts ({top.name}):")
    if top.relevant_excerpts:
        for excerpt in top.relevant_excerpts[:2]:
            lines.append(f"  • {format_excerpt_for_display(excerpt, max_len=220)}")
    else:
        lines.append("  • N/A")
    body = "\n".join(lines)
    if prefix:
        body = f"{prefix}{body}"
    return {"last_response": body}


def interview_questions_node(state: dict) -> dict:
    st = AgentState.model_validate(state)
    user_msg = st.last_user_message or ""
    shortlist, _prefix = resolve_followup_shortlist(st, user_msg)
    if not shortlist:
        return {"last_response": no_shortlist_message(st, action="generate questions for")}
    target = parse_question_target(user_msg, shortlist)
    if isinstance(target, str):
        return {"last_response": target}
    if target is None:
        return {
            "last_response": "Please specify a candidate by id, unique name, or 'top candidate'.",
        }
    try:
        questions = generate_interview_questions(target, st.requirements)
    except ValueError as exc:
        return {"last_response": str(exc)}
    lines = [f"Interview questions for {target.name} (id: {target.candidate_id}):"]
    lines.extend([f"- {q}" for q in questions])
    return {"last_response": "\n".join(lines)}


def candidate_details_node(state: dict) -> dict:
    st = AgentState.model_validate(state)
    user_msg = st.last_user_message or ""
    shortlist, prefix = resolve_followup_shortlist(st, user_msg)
    if not shortlist:
        return {"last_response": no_shortlist_message(st, action="show details for")}

    count = parse_top_count(user_msg, default=1)
    selected = shortlist[:count]
    lines = [f"Detailed report — top {len(selected)} eligible candidate(s):"]
    for c in selected:
        city = c.city if c.city else "Not available in resume data"
        lines.append("")
        lines.append(f"## {c.name} (id: {c.candidate_id})")
        lines.append(f"- City: {city}")
        lines.append(f"- Scores: combined={c.combined_score}, match={c.match_score}, fit={c.fit_score}")
        lines.append(f"- Eligibility: {c.eligibility} | Recommendation: {c.recommendation}")
        lines.append(f"- Round-2 summary: {c.round2_summary or 'N/A'}")
        lines.append(f"- Strengths: {', '.join(c.strengths) or 'N/A'}")
        lines.append(f"- Gaps: {', '.join(c.gaps) or 'N/A'}")
        if c.relevant_excerpts:
            lines.append("- Evidence:")
            for ex in c.relevant_excerpts[:4]:
                lines.append(f"  • {format_excerpt_for_display(ex, max_len=220)}")
        if c.recommendation == "no_hire_for_now" and c.improvement_suggestions:
            lines.append(f"- Improvement suggestions: {'; '.join(c.improvement_suggestions)}")
    body = "\n".join(lines)
    if prefix:
        body = f"{prefix}{body}"
    return {"last_response": body}


def list_corpus_node(state: dict) -> dict:
    from corpus_tools import list_corpus_files

    try:
        files = list_corpus_files()
    except (FileNotFoundError, NotADirectoryError) as exc:
        return {"last_response": str(exc)}
    lines = [f"Corpus files ({len(files)}):"]
    for info in files[:30]:
        lines.append(f"- {info.relative_path} ({info.size_bytes} bytes)")
    if len(files) > 30:
        lines.append(f"... and {len(files) - 30} more")
    return {"last_response": "\n".join(lines)}


def search_files_node(state: dict) -> dict:
    from corpus_tools import search_corpus_filenames

    st = AgentState.model_validate(state)
    msg = st.last_user_message or ""
    query = re.sub(r".*\b(file|filename|resume name)\b", "", msg, flags=re.I).strip(" :")
    if not query:
        query = msg
    try:
        matches = search_corpus_filenames(query)
    except (FileNotFoundError, NotADirectoryError) as exc:
        return {"last_response": str(exc)}
    if not matches:
        return {"last_response": f"No corpus files matched '{query}'."}
    lines = [f"Filename matches for '{query}':"]
    lines.extend([f"- {m}" for m in matches[:25]])
    return {"last_response": "\n".join(lines)}


def read_resume_node(state: dict) -> dict:
    from corpus_tools import read_resume_pdf

    st = AgentState.model_validate(state)
    msg = st.last_user_message or ""
    path_match = re.search(r"read resume\s+(.+)", msg, re.I)
    if not path_match:
        return {"last_response": "Usage: read resume <corpus-relative-path.pdf>"}
    rel = path_match.group(1).strip()
    try:
        doc = read_resume_pdf(rel)
    except (FileNotFoundError, PermissionError) as exc:
        return {"last_response": str(exc)}
    lines = [f"Resume: {doc.relative_path} ({len(doc.pages)} pages)"]
    for page in doc.pages[:3]:
        preview = page.text.strip().replace("\n", " ")[:300]
        lines.append(f"- Page {page.page_number}: {preview}")
    if len(doc.pages) > 3:
        lines.append(f"... {len(doc.pages) - 3} more pages")
    return {"last_response": "\n".join(lines)}


def previous_results_node(state: dict) -> dict:
    st = AgentState.model_validate(state)
    snap = st.last_successful_snapshot if st.request_status == RequestStatus.FAILED else st.previous_snapshot
    if snap is None:
        snap = st.last_successful_snapshot
    if not snap or (not snap.shortlist and not snap.pending_review):
        return {"last_response": "No previous successful results are stored in this session."}
    lines = ["Previous successful results:"]
    for idx, c in enumerate(snap.shortlist[:10], start=1):
        lines.append(f"{idx}. {c.name} (id: {c.candidate_id}) combined={c.combined_score}")
    return {"last_response": "\n".join(lines)}


def fallback_node(state: dict) -> dict:
    return {
        "last_response": (
            "Supported actions: search/refine candidates, compare top N, explain ranking, "
            "interview questions, candidate details, list corpus, search resume filenames, "
            "read resume <path>, or ask for previous results.\n"
            "Example: 'Find me candidates with React and 3+ years experience.'"
        )
    }


def human_feedback_node(state: dict) -> dict:
    st = AgentState.model_validate(state)
    history = list(st.conversation_history)
    if st.last_response:
        history.append(Message(role="assistant", content=st.last_response))
    return {"conversation_history": history}


# =============================================================================
# GRAPH SETUP
# =============================================================================

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("parse_input", parse_input_node)
    graph.add_node("extract_requirements", extract_requirements_node)
    graph.add_node("search_resumes", search_resumes_node)
    graph.add_node("rank_candidates", rank_candidates_node)
    graph.add_node("generate_report", generate_report_node)
    graph.add_node("compare_candidates", compare_candidates_node)
    graph.add_node("explain_ranking", explain_ranking_node)
    graph.add_node("interview_questions", interview_questions_node)
    graph.add_node("candidate_details", candidate_details_node)
    graph.add_node("list_corpus", list_corpus_node)
    graph.add_node("search_files", search_files_node)
    graph.add_node("read_resume", read_resume_node)
    graph.add_node("previous_results", previous_results_node)
    graph.add_node("fallback", fallback_node)
    graph.add_node("human_feedback", human_feedback_node)

    graph.set_entry_point("parse_input")

    def route_from_intent(state: dict) -> str:
        st = AgentState.model_validate(state)
        intent = st.last_user_intent or ""
        if intent in {"search", "refine"}:
            return "extract_requirements"
        route_map = {
            "compare": "compare_candidates",
            "why": "explain_ranking",
            "questions": "interview_questions",
            "details": "candidate_details",
            "list_corpus": "list_corpus",
            "search_files": "search_files",
            "read_resume": "read_resume",
            "previous_results": "previous_results",
            "unsupported": "fallback",
        }
        return route_map.get(intent, "fallback")

    graph.add_conditional_edges(
        "parse_input",
        route_from_intent,
        {
            "extract_requirements": "extract_requirements",
            "compare_candidates": "compare_candidates",
            "explain_ranking": "explain_ranking",
            "interview_questions": "interview_questions",
            "candidate_details": "candidate_details",
            "list_corpus": "list_corpus",
            "search_files": "search_files",
            "read_resume": "read_resume",
            "previous_results": "previous_results",
            "fallback": "fallback",
        },
    )

    def route_after_extract(state: dict) -> str:
        st = AgentState.model_validate(state)
        if st.request_status == RequestStatus.FAILED:
            return "human_feedback"
        return "search_resumes"

    graph.add_conditional_edges(
        "extract_requirements",
        route_after_extract,
        {
            "search_resumes": "search_resumes",
            "human_feedback": "human_feedback",
        },
    )
    graph.add_edge("search_resumes", "rank_candidates")
    graph.add_edge("rank_candidates", "generate_report")

    for node in (
        "generate_report",
        "compare_candidates",
        "explain_ranking",
        "interview_questions",
        "candidate_details",
        "list_corpus",
        "search_files",
        "read_resume",
        "previous_results",
        "fallback",
    ):
        graph.add_edge(node, "human_feedback")

    graph.add_edge("human_feedback", END)

    return graph.compile()


# =============================================================================
# CLI
# =============================================================================

_ANSI_ESCAPE_RE = re.compile(r"\033\[[0-9;]*m")


class TerminalUi:
    """ANSI styling for interactive CLI (respects NO_COLOR and non-TTY stdout)."""

    def __init__(self) -> None:
        self.enabled: bool = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None

    def _paint(self, text: str, *codes: str) -> str:
        if not self.enabled:
            return text
        return f"\033[{';'.join(codes)}m{text}\033[0m"

    def user_prompt(self) -> str:
        return f"\n{self._paint('You: ', '1', '36')}"

    def print_user_turn(self, message: str) -> None:
        if not message:
            return
        label = self._paint("You", "1", "36")
        print(f"{label}: {message}")

    def print_agent(self, message: str) -> None:
        label = self._paint("Agent", "1", "34")
        print(f"\n{label}:")
        print(message)

    def muted_hint(self, text: str) -> str:
        return self._paint(text, "2")

    def format_status(self, message: str) -> str:
        if not self.enabled:
            return f"Status: {message}"
        tag = self._paint("Status", "2", "33")
        detail = self._paint(message, "33")
        return f"{tag}: {detail}"

    @staticmethod
    def pad_line(line: str, width: int = 90) -> str:
        visible = len(_ANSI_ESCAPE_RE.sub("", line))
        return line + " " * max(0, width - visible)


_cli_ui = TerminalUi()


def run_cli() -> None:
    graph = build_graph()
    state = AgentState()
    configure_cli_output()
    print("Matching Agent CLI. Type 'exit' to quit.")
    if _cli_ui.enabled:
        print(
            _cli_ui.muted_hint(
                "(Colors: You=cyan, Status=yellow, Agent=blue — export NO_COLOR=1 to disable)"
            )
        )

    while True:
        user_msg = input(f"{_cli_ui.user_prompt()}").strip()
        if not user_msg:
            continue
        if user_msg.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        state = invoke_with_node_trace(graph, state, user_msg)
        if state.last_response:
            _cli_ui.print_agent(state.last_response)


def run_single_query(query: str) -> None:
    graph = build_graph()
    configure_cli_output()
    state = AgentState()
    _cli_ui.print_user_turn(query)
    state = invoke_with_node_trace(graph, state, query)
    if state.last_response:
        _cli_ui.print_agent(state.last_response)


def configure_cli_output() -> None:
    logging.getLogger().setLevel(logging.WARNING)
    for logger_name in [
        "httpx",
        "httpcore",
        "openai",
        "langchain",
        "langchain_core",
        "langsmith",
        "chromadb",
    ]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
    warnings.filterwarnings(
        "ignore",
        message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater.",
        category=UserWarning,
    )


def prettify_node_name(node_name: str) -> str:
    return node_name.replace("_", " ").title()


NODE_STATUS_LABELS: dict[str, str] = {
    "parse_input": "Understanding your request...",
    "extract_requirements": "Extracting requirements...",
    "search_resumes": "Searching resumes...",
    "rank_candidates": "Reasoning about candidate fit...",
    "generate_report": "Preparing results...",
    "compare_candidates": "Comparing candidates...",
    "explain_ranking": "Explaining ranking decisions...",
    "interview_questions": "Generating interview questions...",
    "candidate_details": "Collecting candidate details...",
    "list_corpus": "Listing corpus files...",
    "search_files": "Searching resume filenames...",
    "read_resume": "Reading resume...",
    "previous_results": "Loading previous results...",
    "human_feedback": "Finalizing response...",
    "fallback": "Clarifying your request...",
}


def show_live_status(message: str, final: bool = False) -> None:
    line = _cli_ui.pad_line(_cli_ui.format_status(message))
    if final:
        print(f"\r{line}")
    else:
        print(f"\r{line}", end="", flush=True)


def invoke_with_node_trace(graph, state: AgentState, user_msg: str) -> AgentState:
    payload = {**state.model_dump(), "last_user_message": user_msg}
    merged = dict(payload)
    last_status = ""
    show_live_status("Thinking...")
    try:
        for event in graph.stream(payload, stream_mode="updates"):
            for node_name, updates in event.items():
                if node_name.startswith("__"):
                    continue
                status = NODE_STATUS_LABELS.get(node_name, f"Working on {prettify_node_name(node_name)}...")
                if status != last_status:
                    show_live_status(status)
                    last_status = status
                merged.update(updates)
        show_live_status("Done.", final=True)
    except Exception as exc:
        logger = logging.getLogger(__name__)
        logger.exception("Graph invocation failed: %s", exc)
        merged["last_response"] = (
            "I could not complete that request due to an analysis or retrieval error. "
            "Your previous successful results remain available via 'show previous results'."
        )
        merged["request_status"] = RequestStatus.FAILED
        show_live_status("I hit a processing issue.", final=True)
    return AgentState.model_validate(merged)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_single_query(" ".join(sys.argv[1:]))
    else:
        run_cli()
