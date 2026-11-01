"""Documented conversations through the real graph and matcher; external I/O is stubbed."""
import pytest
from agent_tools import RequirementSpec, CandidateAnalysis
from domain_models import CandidateProfile, CandidateFact, ExperienceFact, ExtractionStatus
from matching_agent import AgentState, build_graph

@pytest.fixture
def conversation(monkeypatch):
    import matching_agent as agent
    import job_matcher
    monkeypatch.setattr(job_matcher, 'require_openai_key', lambda: None)
    def extract(jd):
        skill = 'data engineering' if 'data engineering' in jd else 'React'
        return RequirementSpec(must_have=[skill], years_experience=4 if skill != 'React' else 3)
    monkeypatch.setattr(agent, 'extract_requirements', extract)
    rows = []
    for i in range(100):
        facts = [CandidateFact(kind='skill',claim=skill,excerpt=skill,page=1) for skill in ['React','data engineering']]
        facts.append(ExperienceFact(kind='experience',years=4,excerpt='4 years experience',page=1))
        profile = CandidateProfile(candidate_id=str(i),source_path=f'{i}.pdf',content_hash=str(i),facts=facts,extraction_status=ExtractionStatus.SUCCESS)
        rows.append({'document':'React and data engineering, 4 years experience', 'distance':i/1000,
                     'metadata':{'path':f'{i}.pdf','name':f'Person {i}','profile':profile.model_dump_json(),'education':''}})
    monkeypatch.setattr(job_matcher, '_semantic_search_all_chunks', lambda *a, **k: rows)
    monkeypatch.setattr(agent, 'analyze_candidate_fit', lambda *a: CandidateAnalysis(summary='Cited fit', strengths=['React'],gaps=[],fit_score=90))
    monkeypatch.setattr(agent, 'generate_interview_questions', lambda *a: [f'Question {i}?' for i in range(5)])
    graph = build_graph()
    state = AgentState()
    def turn(message):
        nonlocal state
        state = AgentState.model_validate(graph.invoke({**state.model_dump(),'last_user_message':message}))
        return state
    return turn


def test_flows_1_through_8_and_10(conversation):
    turn = conversation
    first = turn('Find me candidates with React and 3+ years experience.')
    assert len(first.shortlist) == 10
    assert first.requirements.minimum_experience_years == 3
    assert len({c.candidate_id for c in first.shortlist}) == 10
    assert '| Rank |' in turn('Compare top 3').last_response
    a,b = first.shortlist[:2]
    assert a.name in turn(f'Why did {a.name} rank higher than {b.name}?').last_response
    assert 'could not find' in turn('Why did Nobody rank higher than Missing?').last_response
    assert turn('Generate interview questions for the top candidate').last_response.count('Question ') == 5
    assert 'Round-2 summary' in turn('Show details for top 1').last_response
    refined = turn('Actually require 5+ years and AWS')
    assert refined.requirements.minimum_experience_years == 5
    assert 'aws' in [s.lower() for s in refined.requirements.must_have_labels()]
    assert not refined.shortlist
    assert 'eligible ranked: 0' in turn('Compare top 3').last_response
    assert 'previous eligible shortlist' in turn('Compare previous top 3').last_response
    result = turn('Find candidates with data engineering and 4+ years experience')
    assert len(result.shortlist) == 10
    unchanged = turn('Tell me a joke about hiring')
    assert unchanged.shortlist == result.shortlist
    assert unchanged.requirements == result.requirements
    assert 'Supported actions' in unchanged.last_response


def test_flow_9_corpus_tools(monkeypatch):
    import corpus_tools as corpus
    monkeypatch.setattr(corpus,'list_corpus_files',lambda: [corpus.CorpusFileInfo(relative_path='react.pdf',name='react.pdf',size_bytes=10)])
    monkeypatch.setattr(corpus,'search_corpus_filenames',lambda query: ['react.pdf'])
    monkeypatch.setattr(corpus,'read_resume_pdf',lambda path: corpus.CorpusDocument(relative_path=path,pages=[corpus.CorpusPage(page_number=1,text='React developer')]))
    graph = build_graph()
    for message, expected in [('List corpus','react.pdf'),('Find file react developer','react.pdf'),('read resume react.pdf','Page 1: React developer')]:
        result = graph.invoke({**AgentState().model_dump(),'last_user_message':message})
        assert expected in result['last_response']


def test_old_index_requires_rebuild():
    from job_matcher import validate_index_metadata, IndexRebuildRequired
    with pytest.raises(IndexRebuildRequired, match='python resume_rag.py Resume'):
        validate_index_metadata({'path':'old.pdf'})
