# Conversation Flows

Manual smoke script for `python matching_agent.py`. The CLI colors **You** (cyan), **Status** (yellow), and **Agent** (blue). Use candidate names and ids from your corpus.

---

## Flow 1: Initial search (React + 3+ years)

**User:** Find me candidates with React and 3+ years experience.

**Status:** Thinking… → Understanding… → Extracting requirements… → Searching resumes… → Reasoning… → Preparing results… → Done.

**Agent:** Screening header with `Eligible ranked` / `Pending review` counts; effective requirements; ranked eligible list (ids, scores, recommendations); separate pending-review list with `needs evidence:`; follow-up hints.

**Verify:** Search intent; 3+ years and React in effective requirements; pending rows not mixed into eligible ranking; stable ids; eligible sorted by `combined` descending.

---

## Flow 2: Refinement (5+ years + AWS)

**User:** Actually require 5+ years and AWS.

**Status:** Full search pipeline (refine → extract → search → rank → report).

**Agent:** Updated effective requirements (5+ years, React, AWS); ranking-change section when a prior snapshot exists; eligible count may be 0 with pending-review still populated.

**Verify:** Refine intent; years replace 3→5; AWS added; stale eligible profiles dropped; see Flow 7 if eligible is empty.

---

## Flow 3: Compare top 3

**User:** Compare top 3.

**Status:** Comparing candidates… → Done.

**Agent:** Side-by-side comparison of three candidates from the **eligible shortlist** only.

**Verify:** Needs non-empty eligible shortlist; pending-review candidates are not compared.

---

## Flow 4: Explain ranking (named pair)

**User:** Why did \<name A\> rank higher than \<name B\>?

**Status:** Explaining ranking decisions… → Done.

**Agent:** Score and evidence explanation for both names on the current shortlist.

**Verify:** Real corpus names or ids; ambiguous/unknown names → clarification, no silent substitute.

---

## Flow 5: Interview questions

**User:** Generate interview questions for the top candidate.

**Status:** Generating interview questions… → Done.

**Agent:** 5–7 tailored screening questions.

**Verify:** Resolves top candidate; unknown target → explicit error.

---

## Flow 6: Candidate details

**User:** Show details for top 1.

**Status:** Collecting candidate details… → Done.

**Agent:** Detailed report for one shortlisted candidate.

**Verify:** Details intent, not search (e.g. “Find React candidates in Boston” stays search).

---

## Flow 7: Empty eligible shortlist

**User:** (After Flow 2 with `Eligible ranked: 0`) Compare top 3.

**Status:** Comparing candidates… → Done.

**Agent:** Explains eligible ranked is 0; may mention pending review and suggest `compare previous top 3` if a prior search had eligible candidates.

**Verify:** Must not show prior eligible rows as *current* results; `compare previous top 3` uses `previous_snapshot` when current shortlist is empty.

---

## Flow 8: Multi-round screening (corpus scale)

**User:** Find candidates with data engineering and 4+ years experience.

**Status:** Full search pipeline.

**Agent:** Same report shape as Flow 1; screening recommendations require human judgment.

**Verify:** ≥100 indexed resumes; up to 10 distinct eligible ids when metadata supports it.

---

## Flow 9: Corpus tools (read-only)

**User:** List corpus.  
**Agent:** Bounded corpus file list.

**User:** Find file react developer.  
**Agent:** Filename matches.

**User:** read resume \<corpus-relative-path.pdf\>  
**Agent:** Page previews for path.

**Verify:** Corpus-bound paths; missing file → clear error.

---

## Flow 10: Unsupported command (with active JD)

**User:** Tell me a joke about hiring.

**Status:** Clarifying your request… → Done.

**Agent:** Supported actions list; requirements and shortlist unchanged.

**Verify:** Fallback / unsupported routing only.
