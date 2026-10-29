from pathlib import Path
import pytest
from eligibility import assess_eligibility
from domain_models import requirement_set_from_spec
from corpus_tools import _is_safe_path
from requirements_engine import apply_refinement
from matching_agent import infer_intent

@pytest.mark.parametrize('skill,required', [('C','React'),('JavaScript','Java'),('AWS','AWS certification')])
def test_substrings_do_not_establish_eligibility(skill, required):
    req = requirement_set_from_spec([required], [], None)
    status, _ = assess_eligibility({'skills': skill}, req)
    assert status.value == 'pending_review'

@pytest.mark.parametrize('command', ['remove AWS', 'AWS is optional', 'add Python'])
def test_refinement_intents(command):
    assert infer_intent(command, True) == 'refine'

@pytest.mark.parametrize('command', ['require 5 years and Python', 'require 5+ years and Python'])
def test_years_and_non_aws_skill(command):
    req = requirement_set_from_spec(['React'], [], 3)
    result = apply_refinement(req, command)
    assert result.requirements.minimum_experience_years == 5
    assert result.requirements.must_have_labels() == ['React', 'python']

def test_sibling_directory_is_outside_corpus():
    assert not _is_safe_path(Path('/tmp/Resume-private/file.pdf'), Path('/tmp/Resume'))

def test_ambiguous_refinement_preserves_current_results(graph, mock_pipeline, sample_shortlist, sample_requirements):
    from matching_agent import AgentState
    state = AgentState(job_description='React 3 years', requirements=sample_requirements, shortlist=sample_shortlist)
    result = AgentState.model_validate(graph.invoke({**state.model_dump(), 'last_user_message': 'Actually change things'}))
    assert result.shortlist == sample_shortlist
    assert result.request_status.value == 'failed'

def test_empty_result_preserves_previous_snapshot(graph, mock_pipeline, sample_shortlist, sample_requirements):
    from matching_agent import AgentState, ResultSnapshot
    snapshot = ResultSnapshot(requirements=sample_requirements, shortlist=sample_shortlist)
    state = AgentState(job_description='old', requirements=sample_requirements, shortlist=sample_shortlist, last_successful_snapshot=snapshot)
    result = AgentState.model_validate(graph.invoke({**state.model_dump(), 'last_user_message': 'Find NO_MATCH'}))
    assert result.shortlist == []
    assert result.previous_snapshot == snapshot

def test_analysis_receives_effective_requirements(monkeypatch):
    import matching_agent as agent
    from agent_tools import CandidateAnalysis
    captured = []
    def analyze(jd, candidate):
        captured.append(jd)
        return CandidateAnalysis(summary='fit', fit_score=90)
    monkeypatch.setattr(agent, 'analyze_candidate_fit', analyze)
    agent._shortlist_item_from_pool({'candidate_id':'a', 'candidate_name':'A', 'resume_path':'a.pdf', 'match_score':90}, 'Original React 3 years', requirement_set_from_spec(['Python'], [], 5))
    assert 'Python' in captured[0] and '5+ years' in captured[0]
    assert 'Original' not in captured[0]

def test_invalid_page_citation_fails(monkeypatch, tmp_path):
    import profile_extraction as extraction
    from types import SimpleNamespace
    monkeypatch.setattr(extraction, 'require_openai_key', lambda: None)
    monkeypatch.setattr(extraction, '_page_labelled_text', lambda *args: ('[Page 1] Python', [(1, 'Python')]))
    monkeypatch.setattr(extraction, 'load_cached_profile', lambda *args: None)
    model = SimpleNamespace(invoke=lambda _: SimpleNamespace(content='{"facts":[{"kind":"skill","claim":"Python","excerpt":"Python","page":99}]}'))
    monkeypatch.setattr(extraction, 'ChatOpenAI', lambda **kwargs: model)
    with pytest.raises(ValueError, match='Invalid evidence citation'):
        extraction.extract_profile('a.pdf', tmp_path)
