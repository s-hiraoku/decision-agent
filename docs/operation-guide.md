# Decision Agent Operation Guide

This guide describes how to run Decision Agent so it gradually becomes closer to
the user's judgment.

The operating loop has two jobs:

1. Capture judgment history from real review work.
2. Evaluate whether the agent's judgment is getting closer to the user's judgment.

## Files

Use separate files for profile state, raw history, and fixed evaluation cases.

```text
profiles/
  default.json

records/
  blog_outline.jsonl
  talk_outline.jsonl
  video_script.jsonl

cases/
  blog_outline_cases.jsonl
```

- `profiles/*.json` is the current judgment profile.
- `records/*.jsonl` is append-only operational history.
- `cases/*.jsonl` is a fixed evaluation set used to measure improvement.

Do not treat `records` and `cases` as the same thing. Records are the work log.
Cases are the test set.
Malformed evaluation case rows fail fast because a truncated test set would make
accuracy numbers misleading.

## 1. Review An Artifact

Run a review before showing the user judgment to the agent.

```bash
PYTHONPATH=src python -m decision_agent.cli review \
  profiles/default.json \
  requests/blog-outline-request.json \
  --records records/blog_outline.jsonl
```

The output contains:

- `verdict`: `accept`, `revise`, or `reject`
- `issues`: reasons and suggestions
- `revision_instruction`: the next direction to send back into the loop
- `learned_signals`: which stored rules or records influenced the review

## 2. Capture User Judgment

Write feedback as explicit JSON. Keep it direct. The first implementation learns
best when the user gives concrete fields rather than vague prose.

```json
{
  "verdict": "revise",
  "notes": "The outline explains the concept before the reader feels the problem.",
  "core_issues": [
    "concrete pain point is missing before concept explanation"
  ],
  "revision_direction": "Start with a concrete failure case where an agent creates plausible but misaligned output.",
  "preference_rules": [
    "start with a concrete failure case before naming the concept"
  ],
  "negative_patterns": [
    "concept definition before user pain"
  ]
}
```

Use `core_issues` for what the agent should have noticed. Use
`revision_direction` for what the agent should have instructed next. Use
`preference_rules`, `negative_patterns`, and `positive_examples` only when the
feedback should become durable profile state.

## 3. Iterate

Run one full iteration to review, learn from feedback, update the profile, and
append the raw record.

```bash
PYTHONPATH=src python -m decision_agent.cli iterate \
  profiles/default.json \
  requests/blog-outline-request.json \
  --feedback feedback/blog-outline-feedback.json \
  --records records/blog_outline.jsonl \
  --output profiles/default.json
```

This command appends a `DecisionRecord` to JSONL and writes the updated profile.
The record is raw evidence. The profile is the current summary.

## 4. Evaluate

After collecting several examples, evaluate the agent against fixed cases.

```bash
PYTHONPATH=src python -m decision_agent.cli evaluate \
  profiles/default.json \
  cases/blog_outline_cases.jsonl \
  --records records/blog_outline.jsonl
```

The report includes:

- `verdict_accuracy`: how often `accept` / `revise` / `reject` matches
- `core_issue_accuracy`: how often the agent notices the user's core issues
- `revision_direction_accuracy`: how often the suggested direction matches
- `common_misses`: recurring issues the agent fails to notice
- `suggested_profile_updates`: candidate rules to add to the profile

Do not automatically apply all suggested profile updates. Review them and add
only the rules that actually represent the user's judgment.

## 5. Improve The Profile

When evaluation shows repeated misses, update the profile intentionally.

Good profile updates are specific:

- `start with a concrete failure case before naming the concept`
- `technical blog outlines should include implementation-oriented examples`
- `do not accept concept-first outlines when the reader pain is not visible`

Weak profile updates are too broad:

- `make it better`
- `be more concrete`
- `write like me`

Prefer specific, observable rules that can affect the next review.

## Operating Rhythm

A practical rhythm:

1. Review one artifact.
2. Capture user judgment.
3. Run `iterate`.
4. Add the case to `cases/*.jsonl` if it represents an important judgment.
5. Run `evaluate` after every 5 to 10 new cases.
6. Add only approved profile updates.
7. Repeat.

The goal is not to make the agent confident quickly. The goal is to make judgment
deltas durable, measurable, and reusable.
