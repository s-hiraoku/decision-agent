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
stored natural-language rules, known mistakes, and same-task history. Feedback is
preserved as append-only JSONL records so later iterations can become more
user-aligned.

See [docs/operation-guide.md](docs/operation-guide.md) for the intended operating
loop: review, capture user judgment, iterate, evaluate, then update the profile
only with rules the user agrees with.

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
