# Decision Agent

Decision Agent is a small Python prototype for imitating a user's decision patterns.
It learns from past decision examples and reviews subjective artifacts with an
explainable judgment.

The current implementation is deterministic and does not require an LLM or an API key.
It is designed as a base that can later be connected to chat, forms, or an LLM-based
interviewer.

## Artifact Review

The main direction is reviewing creative or taste-driven work such as blog
outlines, talk outlines, and video scripts.

A review profile contains:

- natural-language preference rules
- negative patterns
- positive examples
- known mistakes from prior verdict deltas
- previous decision records

Recommended local layout:

```text
profiles/default.json
records/blog_outline.jsonl
cases/blog_outline_cases.jsonl
```

Review an artifact:

```bash
PYTHONPATH=src python -m decision_agent.cli review examples/review-profile.json examples/review-request.json
```

The default review engine is `heuristic`, which is deterministic and does not
require an API key. It can also be selected explicitly:

```bash
PYTHONPATH=src python -m decision_agent.cli review \
  examples/review-profile.json \
  examples/review-request.json \
  --engine heuristic
```

The `llm` engine uses the local `claude` CLI (Claude Code) as a subprocess --
not the `anthropic` Python package. This means it transparently uses whatever
local Claude Code authentication is already configured (a Pro/Max
subscription login or an API key), and requires no new pip dependency:

```bash
PYTHONPATH=src python -m decision_agent.cli review \
  examples/review-profile.json \
  examples/review-request.json \
  --engine llm \
  --model claude-opus-4-8
```

`--engine llm` requires Claude Code (`claude`) installed and authenticated
locally; Decision Agent does not manage credentials itself. `--model` is
passed straight through to the `claude` CLI's `--model` flag. `DECISION_AGENT_ENGINE`
sets the default engine as an environment variable (the `--engine` flag takes
precedence). If the LLM review fails for any reason (missing binary,
auth error, malformed response), the command exits with an error rather than
silently falling back to the heuristic engine -- a caller that asked for
`--engine llm` asked for that engine's judgment specifically.

Support for other vendors' CLIs (Codex, Copilot, etc.) is a deliberate
non-goal for now, pending confirmation that they offer an equivalent
schema-forcing structured-output flag.

List profile rules and patterns:

```bash
PYTHONPATH=src python -m decision_agent.cli rules list examples/review-profile.json --json
```

Profiles accept the legacy string form, but saving a profile now writes
structured rule objects with stable IDs, status, provenance, and hit/miss counts.

New rules and patterns learned from feedback start as `candidate` and do not
yet apply to reviews. A candidate only promotes to `active` once the same
pattern recurs across at least two distinct decision records without
contradiction. `learn` prints a summary of which rules were touched and how
many more distinct occurrences are needed before a candidate activates to
stderr (stdout stays reserved for machine-readable output, and `learn`'s
machine-readable artifact is the `--output` profile file); `iterate` embeds
the same summary under a `"learning"` key in its stdout JSON, since its
stdout is already a structured report. `rules list` also reports an
advisory-only staleness flag for active rules that have not yet caused a
review issue, or that are repeatedly contradicted by user feedback -- this
is never used to auto-retire a rule.

When new feedback conflicts with an established (`active`) rule, pattern, or
known mistake, the conflicting signal does not silently overwrite it. The
existing rule keeps governing reviews unchanged, and the conflict shows up
under a `needs_confirmation` list in `learn`/`iterate` output and in
`rules list`, with a `flagged_reason` of `contradicts_established_rule`.
Resolve it the same way as a candidate: `rules approve` keeps the
established rule and discards the new signal; `rules reject` accepts the
new signal and overwrites the established rule's correction. This never
blocks `learn`/`iterate` -- both remain fully non-interactive, and an
unresolved conflict just persists in the profile until a human runs
`rules approve`/`reject`, however long that takes.

Separately, when the agent's own confidence was already low and its
verdict disagreed with the user's feedback, that decision record is
annotated with `flagged_reason: low_confidence_disagreement` -- informational
only; recording and promotion proceed exactly as they would otherwise.

Candidate rules can be promoted or removed without editing JSON by hand:

```bash
PYTHONPATH=src python -m decision_agent.cli rules approve profiles/default.json rule-...
PYTHONPATH=src python -m decision_agent.cli rules reject profiles/default.json rule-...
PYTHONPATH=src python -m decision_agent.cli rules retire profiles/default.json rule-...
```

Move legacy embedded profile history into JSONL:

```bash
PYTHONPATH=src python -m decision_agent.cli migrate-history \
  profiles/old-profile.json \
  --records records/blog_outline.jsonl
```

The migration path reads raw `decision_records` from old profile JSON before the
normal profile model drops embedded history, and JSONL appends skip duplicate
logical records so reruns are safe.

Review with past records:

```bash
PYTHONPATH=src python -m decision_agent.cli review \
  examples/review-profile.json \
  examples/review-request.json \
  --records /tmp/decision-agent-records/blog_outline.jsonl
```

Record feedback and update the profile:

```bash
PYTHONPATH=src python -m decision_agent.cli review examples/review-profile.json examples/review-request.json > /tmp/review.json
PYTHONPATH=src python -m decision_agent.cli learn \
  examples/review-profile.json \
  examples/review-request.json \
  /tmp/review.json \
  examples/review-feedback.json \
  --records /tmp/decision-agent-records/blog_outline.jsonl \
  --output /tmp/learned-profile.json
```

Run one full iteration:

```bash
PYTHONPATH=src python -m decision_agent.cli iterate \
  examples/review-profile.json \
  examples/review-request.json \
  --feedback examples/review-feedback.json \
  --records /tmp/decision-agent-records/blog_outline.jsonl \
  --output /tmp/learned-profile.json
```

Evaluate whether the agent is getting closer to the user's judgment:

```bash
PYTHONPATH=src python -m decision_agent.cli evaluate \
  examples/review-profile.json \
  examples/blog-outline-cases.jsonl \
  --records /tmp/decision-agent-records/blog_outline.jsonl
```

The review path is intentionally simple for now: it checks the artifact against
stored natural-language rules, known mistakes, and same-task history. The
heuristic matcher uses token containment for English-like text and a dependency-free
character n-gram fallback for Japanese or mixed text. Feedback is preserved as
append-only JSONL records so later iterations can become more user-aligned.
Profile writes use an atomic temp-file replace so interrupted writes do not leave
partially written profile JSON.

See [docs/operation-guide.md](docs/operation-guide.md) for the intended operating
loop: review, capture user judgment, iterate, evaluate, then update the profile
only with rules the user agrees with. The implementation roadmap toward
LLM-backed review, rule extraction, and semantic evaluation is defined in
[docs/detailed-design.md](docs/detailed-design.md).

## Option Ranking

The repository still includes the initial numeric option-ranking prototype.

A profile contains:

- criteria and weights, such as `cost`, `quality`, `speed`, or `risk`
- past decision examples
- alternatives for each example with numeric attributes
- the option the user chose

When deciding, the agent combines:

- weighted criterion scores
- similarity to options the user previously chose
- distance from options the user previously rejected

Attribute values are expected to be on a `0..10` scale. Higher is better.

## Development

Run tests:

```bash
PYTHONPATH=src python -m unittest discover -s tests
```

Run strict type checking (`pyright`, via `uv`):

```bash
uv run pyright
```

Create a demo recommendation:

```bash
PYTHONPATH=src python -m decision_agent.cli decide examples/profile.json examples/request.json
```

Install locally:

```bash
python -m pip install -e .
decision-agent decide examples/profile.json examples/request.json
```

Train a profile from its examples:

```bash
decision-agent train examples/profile.json --output trained-profile.json
```

## Example Request

```json
{
  "context": "Choose a contractor for a production feature",
  "alternatives": [
    {
      "name": "Fast vendor",
      "attributes": {"speed": 9, "quality": 5, "cost": 4, "risk": 4}
    },
    {
      "name": "Reliable vendor",
      "attributes": {"speed": 6, "quality": 9, "cost": 7, "risk": 8}
    }
  ]
}
```

The CLI returns the recommended option, ranked scores, and a compact explanation.
