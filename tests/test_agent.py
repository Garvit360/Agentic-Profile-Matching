"""Offline regression tests plus opt-in live corpus checks."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from agent_tools import RequirementSpec, SearchResultBundle
from domain_models import (
    RequirementSet,
    excerpt_in_source,
    format_excerpt_for_display,
    requirement_set_from_spec,
    stable_candidate_id,
)
from eligibility import assess_eligibility
from matching_agent import (
    AgentState,
    ResultSnapshot,
    build_graph,
    infer_intent,
    parse_compare_targets,
    parse_question_target,
    parse_why_targets,
    resolve_candidate_query,
)
from requirements_engine import apply_refinement


# =============================================================================
# Fixtures
# =============================================================================




# =============================================================================
# Domain models
# =============================================================================


def test_excerpt_validation() -> None:
    source = "Skills: React, AWS\nExperience: 5 years"
    assert excerpt_in_source("React, AWS", source)
    assert not excerpt_in_source("Kubernetes expert", source)


def test_format_excerpt_collapses_pdf_line_breaks() -> None:
    mojibake_dash = "\u00e2\u20ac\u201c"
    raw = f"Consultant\n\n,\n\n05/2017\n\nto\nCurrent\nCompany Name\n{mojibake_dash} City"
    formatted = format_excerpt_for_display(raw, max_len=500)
    assert "\n" not in formatted
    assert "05/2017" in formatted
    assert "to Current" in formatted
    assert "Company Name – City" in formatted


def test_candidate_id_stable_normalized_path() -> None:
    assert stable_candidate_id("data/engineer-1.pdf") == stable_candidate_id("./data/engineer-1.pdf")


def test_stable_candidate_id_from_path() -> None:
    a = stable_candidate_id("Resume/foo.pdf")
    b = stable_candidate_id("Resume/foo.pdf")
    c = stable_candidate_id("Resume/bar.pdf")
    assert a == b
    assert a != c


# =============================================================================
# Eligibility and requirements refinement
# =============================================================================


def test_reactjs_satisfies_react_must_have() -> None:
    req = requirement_set_from_spec(must_have=["React"], nice_to_have=[], years_experience=3)
    meta = {"skills": "JavaScript,ReactJS,HTML", "experience_years": 4}
    status, assessments = assess_eligibility(meta, req)
    assert status.value == "eligible"
    assert any(a.status.value == "met" and a.criterion_id.startswith("skill-") for a in assessments)


def test_all_must_haves_required_not_any_skill() -> None:
    req = requirement_set_from_spec(
        must_have=["React", "AWS"],
        nice_to_have=[],
        years_experience=3,
    )
    meta = {"skills": "React, JavaScript", "experience_years": 5}
    status, _ = assess_eligibility(meta, req)
    assert status.value == "pending_review"


def test_experience_not_met_is_ineligible() -> None:
    req = requirement_set_from_spec(must_have=[], nice_to_have=[], years_experience=5)
    meta = {"skills": "Python", "experience_years": 3}
    status, assessments = assess_eligibility(meta, req)
    assert status.value == "ineligible"
    assert any(a.status.value == "not_met" for a in assessments)


def test_refinement_updates_years() -> None:
    base = requirement_set_from_spec(must_have=["React"], nice_to_have=[], years_experience=3)
    result = apply_refinement(base, "Actually require 5+ years and AWS.")
    assert result.applied is True
    assert result.requirements.minimum_experience_years == 5


def test_refinement_remove_aws() -> None:
    base = requirement_set_from_spec(must_have=["React", "AWS"], nice_to_have=[], years_experience=3)
    result = apply_refinement(base, "Remove AWS")
    assert result.applied is True
    labels = [c.label.lower() for c in result.requirements.must_have]
    assert not any("aws" in label for label in labels)


# =============================================================================
# Intent routing and target resolution
# =============================================================================


def test_compare_top_three_routes_to_compare() -> None:
    assert infer_intent("Compare top 3.", has_jd=True) == "compare"


def test_interview_top_candidate_routes_to_questions() -> None:
    msg = "Generate interview questions for the top candidate."
    assert infer_intent(msg, has_jd=True) == "questions"


def test_location_search_routes_to_search_not_details() -> None:
    msg = "Find React candidates in this location: Boston"
    assert infer_intent(msg, has_jd=False) == "search"


def test_unknown_command_with_jd_is_unsupported() -> None:
    assert infer_intent("Tell me a joke about hiring.", has_jd=True) == "unsupported"


def test_parse_why_targets_full_second_name(sample_shortlist) -> None:
    higher, lower, err = parse_why_targets(
        "Why did Alice Smith rank higher than Jane Doe?",
        sample_shortlist,
    )
    assert err is None
    assert higher is not None and lower is not None
    assert "jane" in lower.name.lower()


def test_unknown_name_returns_clarification(sample_shortlist) -> None:
    resolved = resolve_candidate_query("Unknown Person", sample_shortlist)
    assert resolved is None


def test_ambiguous_partial_name(sample_shortlist) -> None:
    dup_shortlist = [
        *sample_shortlist,
        sample_shortlist[0].model_copy(update={"candidate_id": "alice-2", "name": "Alice Brown"}),
    ]
    resolved = resolve_candidate_query("Alice", dup_shortlist)
    assert isinstance(resolved, str)
    assert "Multiple candidates" in resolved


def test_compare_top_three_ids(sample_shortlist) -> None:
    ids = parse_compare_targets("Compare top 3.", sample_shortlist)
    assert len(ids) == 3
    assert ids[0] == sample_shortlist[0].candidate_id


def test_interview_top_candidate_target(sample_shortlist) -> None:
    target = parse_question_target(
        "Generate interview questions for the top candidate.",
        sample_shortlist,
    )
    assert target == sample_shortlist[0]


# =============================================================================
# Graph state (mocked pipeline)
# =============================================================================


def test_empty_search_clears_previous_shortlist(graph, mock_pipeline) -> None:
    state = AgentState()
    state = AgentState.model_validate(
        graph.invoke({**state.model_dump(), "last_user_message": "Find React developers"})
    )
    assert len(state.shortlist) == 1

    state = AgentState.model_validate(
        graph.invoke({**state.model_dump(), "last_user_message": "Find NO_MATCH candidates"})
    )
    assert state.shortlist == []
    assert "No candidates matched" in (state.last_response or "")


def test_compare_after_search(graph, mock_pipeline, sample_shortlist) -> None:
    state = AgentState(shortlist=sample_shortlist)
    state = AgentState.model_validate(
        graph.invoke({**state.model_dump(), "last_user_message": "Compare top 3."})
    )
    assert "| Rank |" in (state.last_response or "")


def test_compare_empty_shortlist_explains(
    graph, sample_shortlist, sample_requirements: RequirementSet
) -> None:
    state = AgentState(
        job_description="Find React devs",
        requirements=sample_requirements,
        shortlist=[],
        pending_review=[sample_shortlist[0]],
        previous_snapshot=ResultSnapshot(shortlist=sample_shortlist),
    )
    state = AgentState.model_validate(
        graph.invoke({**state.model_dump(), "last_user_message": "Compare top 3."})
    )
    response = state.last_response or ""
    assert "eligible ranked: 0" in response
    assert "compare previous top 3" in response


def test_compare_previous_uses_prior_snapshot(graph, sample_shortlist) -> None:
    state = AgentState(
        job_description="Refined JD",
        shortlist=[],
        previous_snapshot=ResultSnapshot(shortlist=sample_shortlist),
    )
    state = AgentState.model_validate(
        graph.invoke({**state.model_dump(), "last_user_message": "Compare previous top 3."})
    )
    assert "previous eligible shortlist" in (state.last_response or "")
    assert "| Rank |" in (state.last_response or "")


# =============================================================================
# Live corpus (opt-in: OPENAI_API_KEY + chroma_db)
# =============================================================================


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
@pytest.mark.skipif(not Path("chroma_db").exists(), reason="chroma_db not indexed")
def test_live_matcher_returns_structure() -> None:
    from job_matcher import run_job_matcher

    result = run_job_matcher("React developer with 3+ years experience", k=5, fetch_multiplier=10)
    assert "top_matches" in result
    assert isinstance(result["top_matches"], list)
