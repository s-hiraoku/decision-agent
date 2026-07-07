from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any


Attributes = dict[str, float]
SUPPORTED_TASK_TYPES = {"blog_outline", "talk_outline", "video_script"}
SUPPORTED_VERDICTS = {"accept", "revise", "reject"}
SUPPORTED_RULE_STATUSES = {"active", "candidate", "retired"}
PROFILE_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class Alternative:
    name: str
    attributes: Attributes

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Alternative":
        return cls(
            name=str(data["name"]),
            attributes={str(key): float(value) for key, value in data.get("attributes", {}).items()},
        )

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "attributes": dict(self.attributes)}


@dataclass(frozen=True)
class DecisionExample:
    context: str
    alternatives: tuple[Alternative, ...]
    chosen: str
    rejected: tuple[str, ...] = ()
    rationale: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DecisionExample":
        return cls(
            context=str(data.get("context", "")),
            alternatives=tuple(Alternative.from_dict(item) for item in data.get("alternatives", [])),
            chosen=str(data["chosen"]),
            rejected=tuple(str(item) for item in data.get("rejected", ())),
            rationale=str(data.get("rationale", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "context": self.context,
            "alternatives": [item.to_dict() for item in self.alternatives],
            "chosen": self.chosen,
            "rejected": list(self.rejected),
            "rationale": self.rationale,
        }

    def alternative_by_name(self, name: str) -> Alternative | None:
        return next((item for item in self.alternatives if item.name == name), None)


@dataclass(frozen=True)
class PreferenceRule:
    text: str
    id: str = ""
    task_types: tuple[str, ...] = ()
    status: str = "active"
    source: str = "user"
    source_record_id: str = ""
    hit_count: int = 0
    miss_count: int = 0
    created_at: str = ""

    @classmethod
    def from_value(cls, value: Any) -> "PreferenceRule":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(text=value, id=_stable_entry_id("preference_rule", value))
        if not isinstance(value, dict):
            raise TypeError("preference rule must be a string or object")

        text = str(value.get("text", ""))
        status = str(value.get("status", "active"))
        if status not in SUPPORTED_RULE_STATUSES:
            raise ValueError(f"unsupported rule status: {status}")
        return cls(
            text=text,
            id=str(value.get("id") or _stable_entry_id("preference_rule", text)),
            task_types=tuple(str(item) for item in value.get("task_types", [])),
            status=status,
            source=str(value.get("source", "user")),
            source_record_id=str(value.get("source_record_id", "")),
            hit_count=int(value.get("hit_count", 0)),
            miss_count=int(value.get("miss_count", 0)),
            created_at=str(value.get("created_at", "")),
        )

    def applies_to(self, task_type: str) -> bool:
        return self.status == "active" and (not self.task_types or task_type in self.task_types)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id or _stable_entry_id("preference_rule", self.text),
            "text": self.text,
            "task_types": list(self.task_types),
            "status": self.status,
            "source": self.source,
            "source_record_id": self.source_record_id,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "created_at": self.created_at,
        }

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return self.text == other
        if isinstance(other, PreferenceRule):
            return self.to_dict() == other.to_dict()
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self.id, self.text))

    def __str__(self) -> str:
        return self.text


@dataclass(frozen=True)
class PatternEntry:
    text: str
    id: str = ""
    task_types: tuple[str, ...] = ()
    status: str = "active"
    source: str = "user"
    source_record_id: str = ""
    hit_count: int = 0
    miss_count: int = 0
    created_at: str = ""

    @classmethod
    def from_value(cls, value: Any, *, kind: str) -> "PatternEntry":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(text=value, id=_stable_entry_id(kind, value))
        if not isinstance(value, dict):
            raise TypeError(f"{kind} must be a string or object")

        text = str(value.get("text", ""))
        status = str(value.get("status", "active"))
        if status not in SUPPORTED_RULE_STATUSES:
            raise ValueError(f"unsupported rule status: {status}")
        return cls(
            text=text,
            id=str(value.get("id") or _stable_entry_id(kind, text)),
            task_types=tuple(str(item) for item in value.get("task_types", [])),
            status=status,
            source=str(value.get("source", "user")),
            source_record_id=str(value.get("source_record_id", "")),
            hit_count=int(value.get("hit_count", 0)),
            miss_count=int(value.get("miss_count", 0)),
            created_at=str(value.get("created_at", "")),
        )

    def applies_to(self, task_type: str) -> bool:
        return self.status == "active" and (not self.task_types or task_type in self.task_types)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "task_types": list(self.task_types),
            "status": self.status,
            "source": self.source,
            "source_record_id": self.source_record_id,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "created_at": self.created_at,
        }

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return self.text == other
        if isinstance(other, PatternEntry):
            return self.to_dict() == other.to_dict()
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self.id, self.text))

    def __str__(self) -> str:
        return self.text


@dataclass(frozen=True)
class DecisionProfile:
    user_id: str
    criteria: dict[str, float]
    schema_version: int = PROFILE_SCHEMA_VERSION
    examples: tuple[DecisionExample, ...] = ()
    preference_rules: tuple[PreferenceRule, ...] = ()
    negative_patterns: tuple[PatternEntry, ...] = ()
    positive_examples: tuple[PatternEntry, ...] = ()
    known_mistakes: tuple["KnownMistake", ...] = ()
    decision_records: tuple["DecisionRecord", ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "preference_rules",
            tuple(PreferenceRule.from_value(item) for item in self.preference_rules),
        )
        object.__setattr__(
            self,
            "negative_patterns",
            tuple(PatternEntry.from_value(item, kind="negative_pattern") for item in self.negative_patterns),
        )
        object.__setattr__(
            self,
            "positive_examples",
            tuple(PatternEntry.from_value(item, kind="positive_example") for item in self.positive_examples),
        )
        object.__setattr__(self, "schema_version", PROFILE_SCHEMA_VERSION)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DecisionProfile":
        return cls(
            user_id=str(data.get("user_id", "user")),
            criteria={str(key): float(value) for key, value in data.get("criteria", {}).items()},
            schema_version=PROFILE_SCHEMA_VERSION,
            examples=tuple(DecisionExample.from_dict(item) for item in data.get("examples", [])),
            preference_rules=tuple(PreferenceRule.from_value(item) for item in data.get("preference_rules", [])),
            negative_patterns=tuple(
                PatternEntry.from_value(item, kind="negative_pattern") for item in data.get("negative_patterns", [])
            ),
            positive_examples=tuple(
                PatternEntry.from_value(item, kind="positive_example") for item in data.get("positive_examples", [])
            ),
            known_mistakes=tuple(KnownMistake.from_dict(item) for item in data.get("known_mistakes", [])),
            decision_records=tuple(DecisionRecord.from_dict(item) for item in data.get("decision_records", [])),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PROFILE_SCHEMA_VERSION,
            "user_id": self.user_id,
            "criteria": dict(self.criteria),
            "examples": [item.to_dict() for item in self.examples],
            "preference_rules": [item.to_dict() for item in self.preference_rules],
            "negative_patterns": [item.to_dict() for item in self.negative_patterns],
            "positive_examples": [item.to_dict() for item in self.positive_examples],
            "known_mistakes": [item.to_dict() for item in self.known_mistakes],
            "decision_records": [item.to_dict() for item in self.decision_records],
        }


@dataclass(frozen=True)
class DecisionResult:
    recommended: str
    scores: dict[str, float]
    explanations: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "recommended": self.recommended,
            "scores": self.scores,
            "explanations": self.explanations,
        }


@dataclass(frozen=True)
class ArtifactReviewRequest:
    task_type: str
    intent: str
    artifact: str
    context: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArtifactReviewRequest":
        task_type = str(data["task_type"])
        if task_type not in SUPPORTED_TASK_TYPES:
            raise ValueError(f"unsupported task_type: {task_type}")
        return cls(
            task_type=task_type,
            intent=str(data.get("intent", "")),
            artifact=str(data.get("artifact", "")),
            context={str(key): str(value) for key, value in data.get("context", {}).items()},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_type": self.task_type,
            "intent": self.intent,
            "artifact": self.artifact,
            "context": dict(self.context),
        }


@dataclass(frozen=True)
class ReviewIssue:
    severity: str
    reason: str
    suggestion: str
    violated_rule_id: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReviewIssue":
        return cls(
            severity=str(data.get("severity", "medium")),
            reason=str(data.get("reason", "")),
            suggestion=str(data.get("suggestion", "")),
            violated_rule_id=str(data.get("violated_rule_id", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "reason": self.reason,
            "suggestion": self.suggestion,
            "violated_rule_id": self.violated_rule_id,
        }


@dataclass(frozen=True)
class ArtifactReview:
    verdict: str
    confidence: float
    summary: str
    issues: tuple[ReviewIssue, ...] = ()
    revision_instruction: str = ""
    learned_signals: tuple[str, ...] = ()
    engine: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArtifactReview":
        verdict = str(data.get("verdict", "revise"))
        if verdict not in SUPPORTED_VERDICTS:
            raise ValueError(f"unsupported verdict: {verdict}")
        return cls(
            verdict=verdict,
            confidence=float(data.get("confidence", 0.0)),
            summary=str(data.get("summary", "")),
            issues=tuple(ReviewIssue.from_dict(item) for item in data.get("issues", [])),
            revision_instruction=str(data.get("revision_instruction", "")),
            learned_signals=tuple(str(item) for item in data.get("learned_signals", [])),
            engine=str(data.get("engine", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "confidence": self.confidence,
            "summary": self.summary,
            "issues": [item.to_dict() for item in self.issues],
            "revision_instruction": self.revision_instruction,
            "learned_signals": list(self.learned_signals),
            "engine": self.engine,
        }


@dataclass(frozen=True)
class UserFeedback:
    verdict: str
    notes: str = ""
    preference_rules: tuple[str, ...] = ()
    negative_patterns: tuple[str, ...] = ()
    positive_examples: tuple[str, ...] = ()
    core_issues: tuple[str, ...] = ()
    revision_direction: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UserFeedback":
        verdict = str(data.get("verdict", "revise"))
        if verdict not in SUPPORTED_VERDICTS:
            raise ValueError(f"unsupported verdict: {verdict}")
        return cls(
            verdict=verdict,
            notes=str(data.get("notes", "")),
            preference_rules=tuple(str(item) for item in data.get("preference_rules", [])),
            negative_patterns=tuple(str(item) for item in data.get("negative_patterns", [])),
            positive_examples=tuple(str(item) for item in data.get("positive_examples", [])),
            core_issues=_string_tuple(data.get("core_issues", []), field_name="core_issues"),
            revision_direction=str(data.get("revision_direction", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "notes": self.notes,
            "core_issues": list(self.core_issues),
            "revision_direction": self.revision_direction,
            "preference_rules": list(self.preference_rules),
            "negative_patterns": list(self.negative_patterns),
            "positive_examples": list(self.positive_examples),
        }


@dataclass(frozen=True)
class KnownMistake:
    pattern: str
    correction: str
    count: int = 1

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KnownMistake":
        return cls(
            pattern=str(data.get("pattern", "")),
            correction=str(data.get("correction", "")),
            count=int(data.get("count", 1)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern": self.pattern,
            "correction": self.correction,
            "count": self.count,
        }


@dataclass(frozen=True)
class DecisionRecord:
    request: ArtifactReviewRequest
    agent_review: ArtifactReview
    user_feedback: UserFeedback
    delta: str
    id: str = ""
    created_at: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DecisionRecord":
        return cls(
            request=ArtifactReviewRequest.from_dict(data["request"]),
            agent_review=ArtifactReview.from_dict(data["agent_review"]),
            user_feedback=UserFeedback.from_dict(data["user_feedback"]),
            delta=str(data.get("delta", "")),
            id=str(data.get("id", "")),
            created_at=str(data.get("created_at", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "request": self.request.to_dict(),
            "agent_review": self.agent_review.to_dict(),
            "user_feedback": self.user_feedback.to_dict(),
            "delta": self.delta,
            "id": self.id,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class EvaluationCase:
    request: ArtifactReviewRequest
    user_judgment: UserFeedback
    id: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvaluationCase":
        return cls(
            request=ArtifactReviewRequest.from_dict(data["request"]),
            user_judgment=UserFeedback.from_dict(data["user_judgment"]),
            id=str(data.get("id", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "request": self.request.to_dict(),
            "user_judgment": self.user_judgment.to_dict(),
        }


@dataclass(frozen=True)
class EvaluationCaseResult:
    id: str
    agent_verdict: str
    user_verdict: str
    verdict_agreement: bool
    core_issue_agreement: bool | None
    revision_direction_agreement: bool | None
    missed_core_issues: tuple[str, ...] = ()
    suggested_profile_updates: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "agent_verdict": self.agent_verdict,
            "user_verdict": self.user_verdict,
            "verdict_agreement": self.verdict_agreement,
            "core_issue_agreement": self.core_issue_agreement,
            "revision_direction_agreement": self.revision_direction_agreement,
            "missed_core_issues": list(self.missed_core_issues),
            "suggested_profile_updates": list(self.suggested_profile_updates),
        }


@dataclass(frozen=True)
class EvaluationReport:
    cases: int
    verdict_accuracy: float
    core_issue_accuracy: float | None
    revision_direction_accuracy: float | None
    common_misses: tuple[str, ...] = ()
    suggested_profile_updates: tuple[str, ...] = ()
    case_results: tuple[EvaluationCaseResult, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "cases": self.cases,
            "verdict_accuracy": self.verdict_accuracy,
            "core_issue_accuracy": self.core_issue_accuracy,
            "revision_direction_accuracy": self.revision_direction_accuracy,
            "common_misses": list(self.common_misses),
            "suggested_profile_updates": list(self.suggested_profile_updates),
            "case_results": [item.to_dict() for item in self.case_results],
        }


def _string_tuple(value: Any, *, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value)
    raise TypeError(f"{field_name} must be a string, list, or tuple")


def _stable_entry_id(kind: str, text: str) -> str:
    digest = sha256(f"{kind}:{text}".encode("utf-8")).hexdigest()[:12]
    return f"rule-{digest}"
