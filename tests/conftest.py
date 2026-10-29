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

from unittest.mock import patch
from agent_tools import SearchResultBundle

@pytest.fixture
def mock_pipeline(sample_requirements: RequirementSet):
    spec = RequirementSpec(
        must_have=sample_requirements.must_have_labels(),
        nice_to_have=sample_requirements.nice_to_have_labels(),
        years_experience=sample_requirements.minimum_experience_years,
        role_context=sample_requirements.role_context,
    )

    def fake_extract(jd: str) -> RequirementSpec:
        return spec

    eligible = [
        {
            "candidate_id": "alice-1",
            "candidate_name": "Alice",
            "resume_path": "Resume/a.pdf",
            "match_score": 90,
            "matched_skills": ["React"],
            "relevant_excerpts": ["Built React apps for 4 years"],
            "reasoning": "Strong match",
            "eligibility": "eligible",
            "criterion_assessments": [],
        }
    ]

    def fake_search(requirements: RequirementSet, jd: str, k: int = 10) -> SearchResultBundle:
        if "NO_MATCH" in jd:
            return SearchResultBundle(eligible=[], pending_review=[])
        return SearchResultBundle(eligible=eligible, pending_review=[])

    def fake_analyze(jd: str, candidate: dict, retry: bool = True):
        from agent_tools import CandidateAnalysis
        from domain_models import AnalysisStatus

        return CandidateAnalysis(
            summary="Strong frontend fit",
            strengths=["React"],
            gaps=[],
            fit_score=85,
            analysis_status=AnalysisStatus.COMPLETE,
        )

    with (
        patch("matching_agent.extract_requirements", side_effect=fake_extract),
        patch("matching_agent.search_resumes", side_effect=fake_search),
        patch("matching_agent.analyze_candidate_fit", side_effect=fake_analyze),
        patch("matching_agent.requirements_from_jd", return_value=sample_requirements),
    ):
        yield
