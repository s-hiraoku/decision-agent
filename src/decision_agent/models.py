from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


Attributes = dict[str, float]
SUPPORTED_TASK_TYPES = {"blog_outline", "talk_outline", "video_script"}
SUPPORTED_VERDICTS = {"accept", "revise", "reject"}


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
class DecisionProfile:
    user_id: str
    criteria: dict[str, float]
    examples: tuple[DecisionExample, ...] = ()
    preference_rules: tuple[str, ...] = ()
    negative_patterns: tuple[str, ...] = ()
    positive_examples: tuple[str, ...] = ()
    decision_records: tuple["DecisionRecord", ...] = ()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DecisionProfile":
        return cls(
            user_id=str(data.get("user_id", "user")),
            criteria={str(key): float(value) for key, value in data.get("criteria", {}).items()},
            examples=tuple(DecisionExample.from_dict(item) for item in data.get("examples", [])),
            preference_rules=tuple(str(item) for item in data.get("preference_rules", [])),
            negative_patterns=tuple(str(item) for item in data.get("negative_patterns", [])),
            positive_examples=tuple(str(item) for item in data.get("positive_examples", [])),
            decision_records=tuple(DecisionRecord.from_dict(item) for item in data.get("decision_records", [])),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "criteria": dict(self.criteria),
            "examples": [item.to_dict() for item in self.examples],
            "preference_rules": list(self.preference_rules),
            "negative_patterns": list(self.negative_patterns),
            "positive_examples": list(self.positive_examples),
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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReviewIssue":
        return cls(
            severity=str(data.get("severity", "medium")),
            reason=str(data.get("reason", "")),
            suggestion=str(data.get("suggestion", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "reason": self.reason,
            "suggestion": self.suggestion,
        }


@dataclass(frozen=True)
class ArtifactReview:
    verdict: str
    confidence: float
    summary: str
    issues: tuple[ReviewIssue, ...] = ()
    revision_instruction: str = ""
    learned_signals: tuple[str, ...] = ()

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
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "confidence": self.confidence,
            "summary": self.summary,
            "issues": [item.to_dict() for item in self.issues],
            "revision_instruction": self.revision_instruction,
            "learned_signals": list(self.learned_signals),
        }


@dataclass(frozen=True)
class UserFeedback:
    verdict: str
    notes: str = ""
    preference_rules: tuple[str, ...] = ()
    negative_patterns: tuple[str, ...] = ()
    positive_examples: tuple[str, ...] = ()

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
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "notes": self.notes,
            "preference_rules": list(self.preference_rules),
            "negative_patterns": list(self.negative_patterns),
            "positive_examples": list(self.positive_examples),
        }


@dataclass(frozen=True)
class DecisionRecord:
    request: ArtifactReviewRequest
    agent_review: ArtifactReview
    user_feedback: UserFeedback
    delta: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DecisionRecord":
        return cls(
            request=ArtifactReviewRequest.from_dict(data["request"]),
            agent_review=ArtifactReview.from_dict(data["agent_review"]),
            user_feedback=UserFeedback.from_dict(data["user_feedback"]),
            delta=str(data.get("delta", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "request": self.request.to_dict(),
            "agent_review": self.agent_review.to_dict(),
            "user_feedback": self.user_feedback.to_dict(),
            "delta": self.delta,
        }
