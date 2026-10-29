"""Shared fixtures for offline tests."""

from __future__ import annotations

import pytest

from agent_tools import CandidateShortlistItem, RequirementSpec
from domain_models import RequirementSet, requirement_set_from_spec
from matching_agent import AgentState, build_graph


@pytest.fixture
def sample_requirements() -> RequirementSet:
    return requirement_set_from_spec(
        must_have=["React", "AWS"],
        nice_to_have=["TypeScript"],
        years_experience=3,
        role_context="Frontend Engineer",
    )


@pytest.fixture
def sample_shortlist() -> list[CandidateShortlistItem]:
    return [
        CandidateShortlistItem(
            candidate_id="alice-abc123",
            name="Alice Smith",
            resume_path="Resume/alice.pdf",
            match_score=85,
            reasoning="Strong React",
            round2_summary="Great fit",
            strengths=["React", "Leadership"],
            gaps=["AWS depth"],
            fit_score=82,
            combined_score=83,
            recommendation="hire",
            eligibility="eligible",
        ),
        CandidateShortlistItem(
            candidate_id="bob-def456",
            name="Bob Jones",
            resume_path="Resume/bob.pdf",
            match_score=78,
            reasoning="Solid JS",
            round2_summary="Good candidate",
            strengths=["JavaScript"],
            gaps=["React depth"],
            fit_score=70,
            combined_score=75,
            recommendation="borderline",
            eligibility="eligible",
        ),
        CandidateShortlistItem(
            candidate_id="jane-ghi789",
            name="Jane Doe",
            resume_path="Resume/jane.pdf",
            match_score=80,
            reasoning="Full stack",
            round2_summary="Balanced",
            strengths=["React", "Node"],
            gaps=["Years"],
            fit_score=76,
            combined_score=78,
            recommendation="no_hire_for_now",
            eligibility="eligible",
        ),
    ]


@pytest.fixture
def graph():
    return build_graph()
