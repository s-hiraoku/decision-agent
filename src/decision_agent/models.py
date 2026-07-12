from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any, cast


def _get(data: dict[str, Any], key: str, default: object = None) -> object:
    return data.get(key, default)


def _get_list(data: dict[str, Any], key: str) -> list[Any]:
    value = data.get(key, [])
    if not isinstance(value, list):
        raise TypeError(f"{key} must be a list")
    return cast("list[Any]", value)


def _get_dict_list(data: dict[str, Any], key: str) -> list[dict[str, Any]]:
    return [_as_str_dict(item) for item in _get_list(data, key)]


def _get_dict(data: dict[str, Any], key: str) -> dict[str, Any]:
    return _as_str_dict(data.get(key, {}))


def _require_dict(data: dict[str, Any], key: str) -> dict[str, Any]:
    return _as_str_dict(data[key])


def _as_str_dict(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("expected an object")
    typed = cast("dict[Any, Any]", value)
    return {str(k): v for k, v in typed.items()}


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
            attributes={str(key): float(value) for key, value in _get_dict(data, "attributes").items()},
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
            context=str(_get(data, "context", "")),
            alternatives=tuple(Alternative.from_dict(item) for item in _get_dict_list(data, "alternatives")),
            chosen=str(data["chosen"]),
            rejected=tuple(str(item) for item in _get_list(data, "rejected")),
            rationale=str(_get(data, "rationale", "")),
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
    source_record_ids: tuple[str, ...] = ()
    hit_count: int = 0
    miss_count: int = 0
    created_at: str = ""
    last_used_at: str = ""
    flagged_reason: str = ""

    @classmethod
    def from_value(cls, value: Any) -> "PreferenceRule":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(text=value, id=_stable_entry_id("preference_rule", value))
        if not isinstance(value, dict):
            raise TypeError("preference rule must be a string or object")
        value = _as_str_dict(cast("Any", value))

        text = str(_get(value, "text", ""))
        status = str(_get(value, "status", "active"))
        if status not in SUPPORTED_RULE_STATUSES:
            raise ValueError(f"unsupported rule status: {status}")
        return cls(
            text=text,
            id=str(_get(value, "id") or _stable_entry_id("preference_rule", text)),
            task_types=tuple(str(item) for item in _get_list(value, "task_types")),
            status=status,
            source=str(_get(value, "source", "user")),
            source_record_ids=_legacy_source_record_ids(value),
            hit_count=int(value.get("hit_count", 0)),
            miss_count=int(value.get("miss_count", 0)),
            created_at=str(_get(value, "created_at", "")),
            last_used_at=str(_get(value, "last_used_at", "")),
            flagged_reason=str(_get(value, "flagged_reason", "")),
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
            "source_record_ids": list(self.source_record_ids),
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "created_at": self.created_at,
            "last_used_at": self.last_used_at,
            "flagged_reason": self.flagged_reason,
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
    source_record_ids: tuple[str, ...] = ()
    hit_count: int = 0
    miss_count: int = 0
    created_at: str = ""
    last_used_at: str = ""
    flagged_reason: str = ""

    @classmethod
    def from_value(cls, value: Any, *, kind: str) -> "PatternEntry":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(text=value, id=_stable_entry_id(kind, value))
        if not isinstance(value, dict):
            raise TypeError(f"{kind} must be a string or object")
        value = _as_str_dict(cast("Any", value))

        text = str(_get(value, "text", ""))
        status = str(_get(value, "status", "active"))
        if status not in SUPPORTED_RULE_STATUSES:
            raise ValueError(f"unsupported rule status: {status}")
        return cls(
            text=text,
            id=str(_get(value, "id") or _stable_entry_id(kind, text)),
            task_types=tuple(str(item) for item in _get_list(value, "task_types")),
            status=status,
            source=str(_get(value, "source", "user")),
            source_record_ids=_legacy_source_record_ids(value),
            hit_count=int(value.get("hit_count", 0)),
            miss_count=int(value.get("miss_count", 0)),
            created_at=str(_get(value, "created_at", "")),
            last_used_at=str(_get(value, "last_used_at", "")),
            flagged_reason=str(_get(value, "flagged_reason", "")),
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
            "source_record_ids": list(self.source_record_ids),
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "created_at": self.created_at,
            "last_used_at": self.last_used_at,
            "flagged_reason": self.flagged_reason,
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
            user_id=str(_get(data, "user_id", "user")),
            criteria={str(key): float(value) for key, value in _get_dict(data, "criteria").items()},
            schema_version=PROFILE_SCHEMA_VERSION,
            examples=tuple(DecisionExample.from_dict(item) for item in _get_dict_list(data, "examples")),
            preference_rules=tuple(PreferenceRule.from_value(item) for item in _get_list(data, "preference_rules")),
            negative_patterns=tuple(
                PatternEntry.from_value(item, kind="negative_pattern")
                for item in _get_list(data, "negative_patterns")
            ),
            positive_examples=tuple(
                PatternEntry.from_value(item, kind="positive_example")
                for item in _get_list(data, "positive_examples")
            ),
            known_mistakes=tuple(KnownMistake.from_dict(item) for item in _get_dict_list(data, "known_mistakes")),
            decision_records=tuple(
                DecisionRecord.from_dict(item) for item in _get_dict_list(data, "decision_records")
            ),
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
    explanations: dict[str, list[str]] = field(default_factory=lambda: cast("dict[str, list[str]]", {}))

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
    context: dict[str, str] = field(default_factory=lambda: cast("dict[str, str]", {}))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArtifactReviewRequest":
        task_type = str(data["task_type"])
        if task_type not in SUPPORTED_TASK_TYPES:
            raise ValueError(f"unsupported task_type: {task_type}")
        return cls(
            task_type=task_type,
            intent=str(_get(data, "intent", "")),
            artifact=str(_get(data, "artifact", "")),
            context={str(key): str(value) for key, value in _get_dict(data, "context").items()},
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
            severity=str(_get(data, "severity", "medium")),
            reason=str(_get(data, "reason", "")),
            suggestion=str(_get(data, "suggestion", "")),
            violated_rule_id=str(_get(data, "violated_rule_id", "")),
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
        verdict = str(_get(data, "verdict", "revise"))
        if verdict not in SUPPORTED_VERDICTS:
            raise ValueError(f"unsupported verdict: {verdict}")
        return cls(
            verdict=verdict,
            confidence=float(data.get("confidence", 0.0)),
            summary=str(_get(data, "summary", "")),
            issues=tuple(ReviewIssue.from_dict(item) for item in _get_dict_list(data, "issues")),
            revision_instruction=str(_get(data, "revision_instruction", "")),
            learned_signals=tuple(str(item) for item in _get_list(data, "learned_signals")),
            engine=str(_get(data, "engine", "")),
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
        verdict = str(_get(data, "verdict", "revise"))
        if verdict not in SUPPORTED_VERDICTS:
            raise ValueError(f"unsupported verdict: {verdict}")
        return cls(
            verdict=verdict,
            notes=str(_get(data, "notes", "")),
            preference_rules=tuple(str(item) for item in _get_list(data, "preference_rules")),
            negative_patterns=tuple(str(item) for item in _get_list(data, "negative_patterns")),
            positive_examples=tuple(str(item) for item in _get_list(data, "positive_examples")),
            core_issues=_string_tuple(_get(data, "core_issues", []), field_name="core_issues"),
            revision_direction=str(_get(data, "revision_direction", "")),
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
    status: str = "active"
    source_record_ids: tuple[str, ...] = ()
    corrected_verdict: str = ""
    flagged_reason: str = ""
    pending_correction: str = ""
    pending_corrected_verdict: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KnownMistake":
        status = str(_get(data, "status", "active"))
        if status not in SUPPORTED_RULE_STATUSES:
            raise ValueError(f"unsupported rule status: {status}")
        return cls(
            pattern=str(_get(data, "pattern", "")),
            correction=str(_get(data, "correction", "")),
            count=int(data.get("count", 1)),
            status=status,
            source_record_ids=tuple(str(item) for item in _get_list(data, "source_record_ids")),
            corrected_verdict=str(_get(data, "corrected_verdict", "")),
            flagged_reason=str(_get(data, "flagged_reason", "")),
            pending_correction=str(_get(data, "pending_correction", "")),
            pending_corrected_verdict=str(_get(data, "pending_corrected_verdict", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern": self.pattern,
            "correction": self.correction,
            "count": self.count,
            "status": self.status,
            "source_record_ids": list(self.source_record_ids),
            "corrected_verdict": self.corrected_verdict,
            "flagged_reason": self.flagged_reason,
            "pending_correction": self.pending_correction,
            "pending_corrected_verdict": self.pending_corrected_verdict,
        }


@dataclass(frozen=True)
class DecisionRecord:
    request: ArtifactReviewRequest
    agent_review: ArtifactReview
    user_feedback: UserFeedback
    delta: str
    id: str = ""
    created_at: str = ""
    flagged_reason: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DecisionRecord":
        return cls(
            request=ArtifactReviewRequest.from_dict(_require_dict(data, "request")),
            agent_review=ArtifactReview.from_dict(_require_dict(data, "agent_review")),
            user_feedback=UserFeedback.from_dict(_require_dict(data, "user_feedback")),
            delta=str(_get(data, "delta", "")),
            id=str(_get(data, "id", "")),
            created_at=str(_get(data, "created_at", "")),
            flagged_reason=str(_get(data, "flagged_reason", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "request": self.request.to_dict(),
            "agent_review": self.agent_review.to_dict(),
            "user_feedback": self.user_feedback.to_dict(),
            "delta": self.delta,
            "id": self.id,
            "created_at": self.created_at,
            "flagged_reason": self.flagged_reason,
        }


@dataclass(frozen=True)
class EvaluationCase:
    request: ArtifactReviewRequest
    user_judgment: UserFeedback
    id: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvaluationCase":
        return cls(
            request=ArtifactReviewRequest.from_dict(_require_dict(data, "request")),
            user_judgment=UserFeedback.from_dict(_require_dict(data, "user_judgment")),
            id=str(_get(data, "id", "")),
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
        items = cast("list[Any] | tuple[Any, ...]", value)
        return tuple(str(item) for item in items)
    raise TypeError(f"{field_name} must be a string, list, or tuple")


def _stable_entry_id(kind: str, text: str) -> str:
    digest = sha256(f"{kind}:{text}".encode("utf-8")).hexdigest()[:12]
    return f"rule-{digest}"


def _legacy_source_record_ids(value: dict[str, Any]) -> tuple[str, ...]:
    ids = tuple(str(item) for item in _get_list(value, "source_record_ids"))
    if ids:
        return ids
    legacy_single = str(_get(value, "source_record_id", ""))
    return (legacy_single,) if legacy_single else ()
