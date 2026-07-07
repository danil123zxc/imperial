# LLM Council Transcript — Imperial RAG Eval Dataset (Quality-Only Plan)

**Date:** 2026-06-25
**Question:** Is a "quality-only, no new questions" strategy the right way to make the Imperial RAG eval dataset trustworthy — and what's the biggest flaw, risk, or missed opportunity?

> Follow-up to `council-transcript-2026-06-23.md`. Since that session the user added gold context IDs to 12 cite rows, expanded `surface_conflict`/`refuse_if_not_found` to 5 each, and built a 20-row Russian judge-calibration set. This session pressure-tests the *next* round, which the user has scoped to **quality-only (harden the existing 37 questions, add no new questions).**

---

## Framed Question

Imperial RAG — a Russian-language RAG over a company's internal operational docs (warehouse regulations, job descriptions, return/logistics procedures, HR & pricing rules; 103 docs). Wrong answers have operational consequences. The eval:

- `evals/questions.jsonl` = 37 questions: 27 `cite_answer` / 5 `surface_conflict` / 5 `refuse_if_not_found`.
- `evals/russian_judge_calibration.jsonl` = 20 rows, all `cite_answer`, 10 correct / 10 incorrect — used to check the Russian LLM judge tracks human judgment.
- Scoring: deterministic `citation_behavior` + `source_hint_behavior` (substring) + retrieval hit@5/precision@5/ndcg@5 — but retrieval relevance is substring-matched on keyword "source hints", NOT on gold context IDs. Ragas: `faithfulness` + `answer_relevancy` ON by default; `factual_correctness`/`context_recall`/`id_context_recall` available but OFF. `reference_answer` exists on every row but the default metrics don't use it.

**Four diagnosed weaknesses:** (1) reference answers are vague meta-descriptions, not the concrete facts the docs contain (7-day return decision, 10%-below "red flag", named approvers), so correctness is unmeasured and a fluent-but-wrong Russian answer passes; (2) retrieval relevance is keyword-substring-matched not gold-ID-based, and only 12/27 cite rows have `reference_context_ids`; (3) the calibration set is all one behavior with trivial polar-opposite negatives; (4) coverage is 73% happy-path cite, the 5 refusals are trivially out-of-corpus, and ~10 document families have zero questions.

**The LOCKED plan (to pressure-test):** W1 rewrite all 27 cite reference answers into concrete grounded facts; W2 backfill the 15 missing gold context IDs; W3 turn on `factual_correctness`; W4 rebuild calibration quality; W5 tighten source hints to be document-specific; W6 add a corpus-aware validation lint. Explicitly NOT doing: hard in-domain near-miss refusals, fact-extraction questions, entity-disambiguation traps, more conflict cases, or covering the untested document families.

---

## Advisor Responses

### The Contrarian
The plan's fatal assumption is that fixing measurement quality of 37 questions makes the eval trustworthy — it only makes it *accurate about a tiny, biased sample*. A sharper instrument that only looks at the happy path is more dangerous, because it manufactures false confidence; a green dashboard on a smoke test reads as "validated" to whoever ships on it. The biggest flaw is W3 + W1 together: turning on `factual_correctness` and grading against rewritten references, but only on questions where the system already retrieves and answers, measures whether the model paraphrases facts it was *handed*. It says nothing about the failure mode that causes operational harm: confidently answering an in-domain question it should have refused, or surfacing the wrong-entity approver — both live entirely in the excluded questions. The cheapest high-value add isn't a new category — it's the **in-domain near-miss refusal**. Today refusal is "capital of Australia"; real harm is "what's the return window for X?" where X isn't in the 7-day rule. One document family, five rows — the only test that distinguishes a careful RAG from a confident liar. Excluding it is the plan's single worst call. Do W1–W6, but un-lock that one.

### The First Principles Thinker
What is the eval actually *for*? "Will a warehouse worker who trusts this system get hurt by a wrong answer?" The plan optimizes the measuring instrument while leaving the thing being measured nearly untouched. An eval is a *sampling estimator of production risk*; its trustworthiness is dominated by **coverage of the failure distribution**, not by the polish of each probe. 37 questions is a sample size — you cannot fix a sampling problem by making each sample prettier. W1–W6 are entirely fidelity-per-question; none touch coverage. The locked constraint optimizes precision while the dominant error is bias → a **false green**, worse than a smoke test, that gets trusted in exactly the high-consequence cases it never probed. Every named failure mode — wrong-entity, in-domain refusal, partial facts — is a *question type*, not a reference rewrite. W4 adds hard negatives to calibration, conceding the judge needs harder cases, while refusing to give the *system* harder cases — incoherent. Reframe the constraint: not "no new questions" but **"no new document families"** — adversarial variants of the 27 docs you already cover cost almost nothing. Do W1–W6, but bolt on ~15 in-domain near-misses.

### The Expansionist
This is a half-built asset you're refusing to finish. **W1 is question-generation in disguise:** to rewrite 27 references into concrete facts you must mine every gradeable fact — the 7-day window, the 10%-below red flag, the named approvers — and that extraction is 90% of the cost of writing new questions. One grounded fact = one reference answer *or* three questions (cite + conflict + refuse). Same labor, 3x the eval. The corpus is a moat; the eval should **compound with it**, not certify a frozen snapshot of 37 lookups. W1+W2 build precisely the infrastructure — grounded facts plus context IDs — needed to auto-generate and grade questions at scale; W4's calibrated Russian judge is reusable across an unbounded set. You're constructing a question factory, then capping output at 37. The ~10 untested families are production blast zones you're choosing not to instrument. "Trustworthy" and "37 questions" cannot coexist — an honest smoke test is still a smoke test. Do W1–W6 — then don't stop; let every fact spawn its question and pipe the context IDs into synthetic generation. Marginal cost near zero.

### The Outsider
You're treating "no new questions" as a virtue when it's actually the bug. A prior council literally called 37 a smoke test, and the response is to polish the 37? That's like a restaurant failing inspection for having 3 dishes, and the fix being "make those 3 dishes really good." The dishes were never the problem — the menu is. What is this eval FOR? Wrong answers have operational consequences. The 6 fixes make the eval more rigorous about the same 37 happy paths, not more *representative* of what users will actually ask — increasing precision on an unrepresentative sample. The biggest flaw: W1–W6 fix grading, not coverage, and coverage is where the operational risk lives. You explicitly excluded the ~10 untested document families — 10 whole categories a real employee will ask, where you have zero signal. And W1 (rewriting all 27 references) is the expensive, judgment-heavy part; once you're reading every doc closely, writing 2 new questions per family is nearly free. The constraint forbids the cheapest, highest-value addition at the exact moment it costs least.

### The Executor
W1 alone is most of the value, and it's blocked on a step nobody put in the plan. You can't rewrite 27 reference answers from your armchair — each requires opening the source doc, locating the exact fact (7 calendar days, 10% red flag, named approver), and pasting the gold span. That's the same physical motion as W2 (context IDs) and W5 (document-specific hints). So the biggest flaw is **sequencing six workstreams instead of one pass per question**: done together it's one trip that yields the answer, the ID, and the hint simultaneously. Monday morning, first step: take the 12 cite rows that already have `reference_context_ids`, open the doc, extract the fact, write all three fields at once — that's your template, and it de-risks W2's 15 missing IDs. The biggest **risk**: turn on `factual_correctness` (W3) before W1 lands and the judge grades fluent-wrong answers against vague references — a green dashboard that's still lying. **W3 must be gated on W1+W4 — never flip it early.** And front-load W6's lint (assert every `reference_context_id` resolves, every hint substring-hits its doc), then let it police the rewrite.

---

## Peer Reviews

**Anonymization mapping:** A = The Outsider · B = The Executor · C = The Contrarian · D = The Expansionist · E = The First Principles Thinker

### Reviewer 1
1. **Strongest = B** — the only operationally executable answer (one pass per question; the 12 ID-carrying rows as template) and it catches the W3-before-W1 "green dashboard lying" sequencing bug.
2. **Biggest blind spot = A** — pure diagnosis ("the menu is the problem") with no execution path and no specific catching question, while most redundant with C/D/E.
3. **All missed:** W4 (calibration) is the *precondition*, not a peer workstream. Every automated grade is only as trustworthy as the judge; the 20-row set (one behavior, trivial negations) must be validated against human labels with inter-rater agreement before it can certify anything.

### Reviewer 2
1. **Strongest = C** — converts the shared coverage critique into one surgical, buildable move: the in-domain near-miss refusal.
2. **Biggest blind spot = D** — "one fact = three questions, marginal cost near zero" is wrong; conflict/refuse questions need *engineered conditions* (a contradicting doc pair, a verified corpus gap), not just a mined fact. The factory math only holds for cite questions.
3. **All missed:** W3 is only trustworthy if the judge is validated against the calibration set first, so W4 must precede W3. Coverage and judge-validity are the same blocker.

### Reviewer 3
1. **Strongest = E** — sharpest reframe ("no new *document families*") and cleanest contradiction catch on W4.
2. **Biggest blind spot = B** — accepts the lock and optimizes the wrong objective efficiently.
3. **All missed:** nobody grounds coverage in **production query logs** — the real fix for a bias problem is sampling actual user questions to learn the failure distribution, not inventing variants; and no held-out human-graded check to validate the judge itself.

### Reviewer 4
1. **Strongest = C** — names the specific catastrophic failure mode and ties it to the W1+W3 interaction; most surgical, respects the constraint.
2. **Biggest blind spot = B** — tunnel-vision on execution sequencing while accepting the wrong-scoped frame.
3. **All missed:** the 20-row Russian judge-calibration set and the judge itself are load-bearing — if `factual_correctness` runs through a judge calibrated on trivial negations, no reference rewrite makes the *grader* trustworthy.

### Reviewer 5
1. **Strongest = C** — surgical actionability; isolates the single highest-value unlock (E close second on theory).
2. **Biggest blind spot = B** — honors the lock, under-weights coverage / false-green bias.
3. **All missed:** none interrogate whether a judge calibrated on trivial negations can even grade the harder adversarial cases they all propose; also no anchoring of coverage to real production query logs.

---

## Chairman's Verdict

### Where the Council Agrees
- **All five advisors independently reject the premise.** Not one defends "quality-only, no new questions." Five independent angles all land on "the constraint is the bug" — the highest-confidence signal in the room.
- **This is a bias problem, not a precision problem — and W1–W6 fix precision.** Sharpening the instrument on a 73%-happy-path sample produces a *false green*: worse than a smoke test, because the dashboard now reads as validated.
- **W3 (`factual_correctness`) must be gated on W1+W4.** Turning it on early grades fluent-wrong answers against vague references through an uncalibrated judge → manufactured confident green.
- **W1 is the expensive, judgment-heavy core, and it makes the highest-value addition nearly free.** Rewriting a reference = opening the doc and locating the exact fact = the same motion as W2, W5, and writing a new question.

### Where the Council Clashes
- **Scalpel vs. factory.** Contrarian + First Principles: un-lock exactly one thing — ~5–15 in-domain near-miss refusals that separate a careful RAG from a confident liar. Expansionist: don't stop at 37 — W1+W2+W4 are infrastructure to auto-generate at scale. The peer reviewers broke decisively toward the scalpel; Reviewer 2 punctured the factory: the "one fact = three questions" math only holds for cite questions, because conflict/refuse need *engineered conditions*, not mined facts — and those are exactly the behaviors that carry the operational risk.
- **Respect the lock vs. refuse the frame.** The Executor alone honors "quality-only" and is named strongest on executability by every reviewer — yet every reviewer calls that his blind spot ("solving the wrong problem flawlessly"). His sequencing insight survives into the recommendation regardless.

### Blind Spots the Council Caught
- **The judge itself is unvalidated — and this collapses the plan, including the un-lock everyone proposed.** Emerged only in peer review; **all five reviewers flagged it independently.** Every `factual_correctness` score is only as trustworthy as a judge calibrated on trivial polar-opposite negations — and such a judge cannot be trusted to grade the *hard* adversarial cases the advisors themselves prescribe. W4 is therefore the **precondition** for W3 and every un-lock. Correct order: W4 → validate judge vs. human labels (inter-rater agreement) → W3 → harder cases.
- **Nobody anchored coverage to production reality.** The principled fix for a *bias* problem is to sample the real failure distribution from **production query logs**, not to invent cleverer variants. If logs exist, they dominate every proposal here.
- **No held-out human-graded anchor.** Even a recalibrated judge needs a small human-graded ground-truth set, or trust just moves from the references to the calibration set.

### The Recommendation
**Do W1–W6 — none of it is wasted — but change two things or you will ship a green dashboard that lies.**

The honest verdict: **"quality-only, no new questions" is the wrong strategy** (5 advisors, 5 reviewers agree). It optimizes precision when the problem is bias, and forbids the cheapest high-value addition at the exact moment W1 makes it nearly free. The constraint-respecting version:

1. **Reorder around the judge — W4 first, not fourth.** Rebuild the calibration set with hard/borderline negatives, then validate it against a small human-graded held-out set with inter-rater agreement. **W3 stays OFF until that lands.** A polished reference graded by a judge calibrated on "capital of Australia"-grade negations is not trustworthy no matter how concrete the fact.
2. **Crack the lock exactly once — for in-domain near-miss refusals.** Not the factory (the "marginal cost ≈ 0" math fails for conflict/refuse). Add ~5–8 questions like "what's the return window for [scenario not in corpus]?" — in-domain, plausible, *should refuse*. The single test separating a careful RAG from a confident liar, at near-zero cost since you're already reading every doc for W1. Reframe the constraint as **"no new document families,"** not "no new questions."

**Chairman's addition:** before anything, spend 30 minutes checking whether **production query logs exist.** If they do, they outrank every invented question in this thread — a bias problem is solved by sampling the real distribution, not inventing a cleverer sample.

### The One Thing to Do First
**Rebuild and human-validate the judge (W4) before touching anything else** — start by hand-grading a held-out set of ~15 answers and measuring whether your calibrated judge agrees with the humans. Everything downstream (W3, the reference rewrites, the new near-miss questions) produces a number that is only as trustworthy as that judge. Validate the instrument before polishing what it measures.
