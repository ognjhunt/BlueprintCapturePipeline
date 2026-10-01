---
name: deep-research
description: Answers a substantial open question by gathering sources, weighing their quality, resolving disagreements between them, and reporting a conclusion with its evidence and its limits. Use this whenever the user asks you to research, compare options, evaluate a technology choice, find out what is known about something, or asks a question that cannot be answered from one source, including "should we use X or Y" decisions. For condensing material the user already has, use summarization.
license: MIT
---

# Deep research

Research is the disciplined management of your own uncertainty. The output is not a pile of
links — it is a conclusion, the evidence behind it, and an honest account of what would change
it.

Two failure modes dominate, and both feel like success while they happen. **Confirmation**:
finding sources that agree with the first plausible answer and stopping. **Accumulation**:
gathering forty sources and reporting them all, leaving the actual work to the reader.

## 1. Sharpen the question

Most research questions arrive underspecified in ways that change the answer. Before searching,
pin down:

- **The decision behind it.** "Postgres or MongoDB" is unanswerable; "which fits a write-heavy
  event log with strict consistency, on a team that knows SQL" is answerable.
- **The constraints that disqualify.** Budget, existing stack, team skills, compliance,
  timeline. These usually eliminate most of the option space immediately.
- **What would count as an answer.** A recommendation, a comparison table, a number, a yes/no?

If the sharpened question is materially different from the one asked, say so and confirm before
spending the effort.

**Done when:** the question names a decision, its constraints, and the form of the answer.

## 2. Search from several angles

One phrasing returns one neighbourhood of results. Deliberately vary:

- The **vocabulary** — practitioners, vendors, and academics name the same thing differently
- The **stance** — search for the failure mode too: "X problems", "migrating away from X",
  "X postmortem". Critical sources are systematically underrepresented in search results
  because vendors publish more than victims
- The **recency** — what was true three years ago frequently is not; check whether the
  consensus you found has an expiry date

**Done when:** you have found at least one credible source that disagrees with your leading
answer, or established that none exists.

## 3. Weigh sources rather than counting them

Ten blog posts repeating one benchmark are one piece of evidence, not ten. Trace claims to
their origin — a surprising number of confident statements share a single unverified ancestor.

Rank by:
- **Primary over secondary:** documentation, source code, the original paper, the actual
  benchmark over someone's summary of it
- **Interest:** who benefits if you believe this? A vendor comparison is evidence about the
  vendor's positioning, not about the product
- **Specificity:** a source that states its conditions, versions, and hardware is worth more
  than one that reports a number
- **Falsifiability:** could this claim be checked? Claims that cannot be wrong are not
  evidence

**Done when:** each key claim has a traced origin and a noted interest.

## 4. Resolve disagreement instead of averaging it

When sources conflict, that is the most informative moment in the research — do not smooth it
over with "opinions differ".

Usually one of:
- **Different conditions:** both are right about different scales, versions, or workloads.
  This is the most common, and finding the boundary is often the real answer.
- **Different definitions:** they are measuring different things under one word.
- **One is stale:** the behaviour changed in a release.
- **One is wrong:** it happens; say so, with the reason.

**Done when:** each significant conflict is explained, not just reported.

## 5. Report the conclusion first

```markdown
# <Question>

**Answer:** the conclusion, in two or three sentences, up front.
**Confidence:** high / medium / low, and what drives it.

## Why
The reasoning, with evidence attached to each step. [source]

## What would change this
The finding that would flip the conclusion. Be specific.

## Considered and rejected
The alternatives, and the specific reason each lost.

## Open
What is still unknown, and what it would take to resolve.

## Sources
Grouped by weight, with a note on what each contributed.
```

Lead with the answer. A reader who stops after the first paragraph should still have the
conclusion.

## Rules that keep it honest

- **Never cite a source you did not open.** This is the single most important rule here. A
  citation asserts you read it; borrowing one from another article's reference list and passing
  it off as your own reading is fabrication, even when the source turns out to say what you
  claimed.
- **Mark inference as inference.** "The docs do not say, but the behaviour implies" is a
  legitimate and valuable sentence. Presenting it as documented is not.
- **Report the gaps you could not fill.** Absence of evidence is a finding, especially the
  absence of independent benchmarks or of any critical coverage.
- **State confidence separately from the conclusion.** A low-confidence recommendation is
  useful. A high-confidence one that was actually low is a trap.
