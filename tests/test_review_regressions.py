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

def _extract_with_fake_model(monkeypatch, tmp_path, page_text, facts):
    import profile_extraction as extraction
    from types import SimpleNamespace
    model = SimpleNamespace(invoke=lambda _: extraction.ExtractionPayload.model_validate({'facts': facts}))
    model.with_structured_output = lambda *a, **k: model
    monkeypatch.setattr(extraction, 'require_openai_key', lambda: None)
    monkeypatch.setattr(extraction, '_page_labelled_text', lambda *args: ('combined', list(page_text.items())))
    monkeypatch.setattr(extraction, 'load_cached_profile', lambda *args: None)
    monkeypatch.setattr(extraction, 'save_cached_profile', lambda *args: None)
    monkeypatch.setattr(extraction, 'ChatOpenAI', lambda **kwargs: model)
    return extraction.extract_profile('a.pdf', tmp_path)

def test_invalid_page_citation_fails(monkeypatch, tmp_path):
    profile = _extract_with_fake_model(monkeypatch, tmp_path, {1: 'Python'},
        [{'kind':'skill','claim':'Java','excerpt':'Java','page':99}])
    assert profile.extraction_status.value == 'failed'
    assert profile.facts == []

def test_unsupported_facts_are_dropped_supported_facts_kept(monkeypatch, tmp_path):
    page_text = {
        1: 'Skilled professional with \n20\n \nyears in security.\nFirst Aid training (\n 2016\n )',
        2: 'Core competencies include\nSkills\nMaintenance, Corrective Maintenance, Documentation',
    }
    profile = _extract_with_fake_model(monkeypatch, tmp_path, page_text, [
        {'kind':'skill','claim':'First Aid training','excerpt':'First Aid training (2016)','page':1},
        {'kind':'experience','years':20,'excerpt':'Skilled professional with 20 years','page':1},
        {'kind':'skill','claim':'Documentation','excerpt':'Core competencies include Maintenance, Corrective Maintenance, Documentation','page':1},
        {'kind':'experience','years':25,'excerpt':'Skilled professional with 20 years','page':1},
        {'kind':'skill','claim':'React','excerpt':'First Aid training (2016)','page':1},
        {'kind':'skill','claim':'Welding','excerpt':'Core competencies include Welding','page':2},
    ])
    assert profile.extraction_status.value == 'success'
    assert [(f.page, f.excerpt) for f in profile.facts] == [
        (1, 'First Aid training (2016)'),
        (1, 'Skilled professional with 20 years'),
        (2, 'Maintenance, Corrective Maintenance, Documentation'),
    ]

def test_excerpt_check_ignores_pdf_whitespace_but_not_wording():
    from domain_models import excerpt_in_source, term_in_excerpt
    source = 'First Aid training (\n 2016\n )\nÂ Weapons training'
    assert excerpt_in_source('First Aid training (2016)', source)
    assert excerpt_in_source('Weapons training', source)
    assert not excerpt_in_source('First Aid trainings (2016)', source)
    assert not excerpt_in_source('React', 'Reactive maintenance')
    assert excerpt_in_source('C/C++', 'Languages: \nC/\nC++, Verilog')
    assert term_in_excerpt('C', 'Languages: C, Python')
    assert not term_in_excerpt('Java', 'JavaScript')

@pytest.mark.parametrize('years', [20, 1, 24, 8])
def test_experience_has_numeric_years(years):
    from profile_extraction import ExtractionPayload
    payload = ExtractionPayload.model_validate({'facts':[{'kind':'experience','years':years,'excerpt':f'{years} years total professional experience','page':1}]})
    assert payload.facts[0].years == years

@pytest.mark.parametrize('fact', [
    {'kind':'skill','claim':20,'excerpt':'20','page':1},
    {'kind':'experience','years':'20','excerpt':'20','page':1},
    {'kind':'experience','years':-1,'excerpt':'-1','page':1},
])
def test_invalid_fact_types_are_rejected(fact):
    from profile_extraction import ExtractionPayload
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        ExtractionPayload.model_validate({'facts':[fact]})


def test_native_chroma_reads_persisted_profile(tmp_path, monkeypatch):
    import chromadb
    import job_matcher
    from domain_models import CandidateProfile, CandidateFact, ExtractionStatus
    from embedding_utils import CHROMA_COLLECTION_NAME
    from types import SimpleNamespace
    profile = CandidateProfile(candidate_id='a',source_path='a.pdf',content_hash='abc',facts=[CandidateFact(kind='skill',claim='React',excerpt='React',page=1)],extraction_status=ExtractionStatus.SUCCESS)
    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.get_or_create_collection(CHROMA_COLLECTION_NAME)
    collection.upsert(ids=['a'],embeddings=[[1.0,0.0]],documents=['React'],metadatas=[{'path':'a.pdf','profile':profile.model_dump_json()}])
    monkeypatch.setattr(job_matcher,'OpenAIEmbeddings',lambda **kwargs: SimpleNamespace(embed_query=lambda query:[1.0,0.0]))
    rows = job_matcher._semantic_search_all_chunks('React',str(tmp_path),10)
    assert rows[0]['metadata']['path'] == 'a.pdf'
    assert rows[0]['document'] == 'React'
