# Agentic Profile Matching (LangGraph)

LangGraph-based matching agent that ranks resumes against structured requirements, supports iterative refinement, comparison, explanations, and interview questions via a CLI.

## Requirements

- Python 3.12 (recommended)
- OpenAI API key for indexing, search, and analysis

## Setup

```bash
cd Agentic_Profile_Matching
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add OPENAI_API_KEY
```

## Index resumes

```bash
unzip -q Resume.zip -d .
python resume_rag.py Resume
```

Re-running indexing uses deterministic chunk IDs for the same sources to avoid unbounded duplicate growth.

## Run the agent

```bash
python matching_agent.py
```

Single-shot:

```bash
python matching_agent.py "Find me candidates with React and 3+ years experience"
```

## Tests (offline, no API key)

```bash
pytest tests/test_agent.py -q
```

Live corpus smoke (optional, requires indexed `chroma_db` and API key):

```bash
pytest tests/test_agent.py -q -m live
```

## Example prompts

- `Find me candidates with React and 3+ years experience`
- `Actually require 5+ years and AWS`
- `Compare top 3`
- `Why did Alice Smith rank higher than Jane Doe?`
- `Generate interview questions for the top candidate`
- `Show details for top 1`
- `List corpus files`
- `Show previous results`

## Architecture

See `docs/state_machine.mmd` (Mermaid). Export to SVG/PNG with:

```bash
npx -y @mermaid-js/mermaid-cli -i docs/state_machine.mmd -o docs/state_machine.svg
```

## Limitations

- Screening scores are heuristic (60% retrieval / 40% deep fit); not calibrated hiring probabilities.
- Missing skill evidence yields **pending review**, not confirmed absence.
- OCR for scanned PDFs is out of scope.
- Milestone 1 tool parity is not claimed; this repo exposes read-only corpus listing, filename search, and resume reading.

## Recovery

- **Missing vector DB:** run `python resume_rag.py Resume`
- **Stale results in session:** run a fresh search; failed requests keep the last successful snapshot (`show previous results`)
- **API errors:** check `.env` and retry; incomplete deep analysis receives no final hire decision

## Evidence index migration

Indexes created before structured profiles were integrated must be rebuilt with `python resume_rag.py Resume`. Ingestion extracts and caches page-cited facts using the model; this incurs API usage on uncached resumes. Invalid extraction stops ingestion explicitly. Replacement chunks are written before obsolete chunks are deleted. Search requires the profile metadata and does not substitute regex estimates when it is missing.

Run all offline regression tests with `python -m pytest -q`. Live provider validation and the submission video remain separate release checks.
