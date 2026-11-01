# Conversation Flows

Manual smoke script for `python matching_agent.py`. The CLI colors **You** (cyan), **Status** (yellow), and **Agent** (blue). Copy candidate names and ids from the eligible list the CLI just printed.

Record **Flows 1–6** in one fresh session, in this order. SQL with a stated total of 3 or more years is verified on enough resumes that the ranked list stays populated through the years change, so compare, ranking, details, and interview questions all have candidates to use.

```bash
python matching_agent.py
```

Flows 7–10 are extra checks. Leave them out of the recording.

---

## Flow 1: Initial search (SQL + 3+ years)

**User:** Find me candidates with SQL and 3+ years experience.

**Status:** Thinking… → Understanding… → Extracting requirements… → Searching resumes… → Reasoning… → Preparing results… → Done.

**Agent:** Screening header with `Eligible ranked` / `Pending review` counts; effective requirements (3+ years, SQL); numbered eligible list (ids, scores, recommendations); separate pending-review list with `needs evidence:`; follow-up hints.

**Verify:** Search intent; 3+ years and SQL in effective requirements; eligible ranked is at least 3; pending rows are not numbered into the eligible ranking; stable ids; eligible sorted by `combined` descending.

---

## Flow 2: Refinement (5+ years)

**User:** Actually require 5+ years.

**Status:** Full search pipeline (refine → extract → search → rank → report).

**Agent:** Updated effective requirements (5+ years, SQL). Eligible ranked stays at least 3. Ranking-change section notes anyone dropped because their verified total is under 5.

**Verify:** Refine intent; years replace 3→5; SQL remains a must-have; the numbered list is still long enough for Flow 3.

---

## Flow 3: Compare top 3

**User:** Compare top 3.

**Status:** Comparing candidates… → Done.

**Agent:** Side-by-side comparison of three candidates from the **eligible shortlist** only.

**Verify:** The three names are the first three numbered rows from Flow 2. Pending-review candidates are not compared.

---

## Flow 4: Explain ranking (named pair)

**User:** Why did \<name on line 1\> rank higher than \<name on line 2\>?

**Status:** Explaining ranking decisions… → Done.

**Agent:** Score and evidence explanation for both names on the current shortlist.

**Verify:** Use the two names just printed in the eligible list. An unknown name returns a clarification and does not substitute someone else.

---

## Flow 5: Interview questions

**User:** Generate interview questions for the top candidate.

**Status:** Generating interview questions… → Done.

**Agent:** 5–7 tailored screening questions for the first eligible candidate.

**Verify:** Questions name that candidate. An unknown target returns an explicit error.

---

## Flow 6: Candidate details

**User:** Show details for top 1.

**Status:** Collecting candidate details… → Done.

**Agent:** Detailed report for the first eligible candidate (strengths, gaps, suggestions).

**Verify:** Details intent, not a new search (e.g. “Find SQL candidates in Boston” stays a search).

---

## Flow 7: Empty eligible shortlist (do not record)

**User:** (In a separate session, after a search that has eligible candidates) Actually require AWS.

**Status:** Full search pipeline, then a later `Compare top 3` → Comparing candidates… → Done.

**Agent:** AWS is added and eligible ranked drops to 0, because that skill is not a verified quote on the retrieved resumes. `Compare top 3` explains that eligible ranked is 0 and that compare uses only the eligible list. `Compare previous top 3` uses the previous eligible snapshot.

**Verify:** Current results must not reprint the previous eligible rows. `compare previous top 3` reads `previous_snapshot` only when the current shortlist is empty.

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

**User:** Find file sql developer.  
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
