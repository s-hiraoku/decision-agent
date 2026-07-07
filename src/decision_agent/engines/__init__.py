from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from decision_agent.models import ArtifactReview, ArtifactReviewRequest, DecisionProfile, DecisionRecord, UserFeedback


class ReviewEngine(Protocol):
    name: str

    def review(
        self,
        request: ArtifactReviewRequest,
        profile: DecisionProfile,
        records: tuple[DecisionRecord, ...],
    ) -> ArtifactReview: ...


class FeedbackExtractor(Protocol):
    def extract(
        self,
        request: ArtifactReviewRequest,
        agent_review: ArtifactReview,
        user_feedback: UserFeedback,
        profile: DecisionProfile,
    ) -> "RuleProposalSet": ...


class AgreementJudge(Protocol):
    name: str

    def judge(self, expected: UserFeedback, review: ArtifactReview) -> "AgreementJudgment": ...


@dataclass(frozen=True)
class CoreIssueJudgment:
    issue: str
    noticed: bool
    evidence: str = ""


@dataclass(frozen=True)
class AgreementJudgment:
    core_issues: tuple[CoreIssueJudgment, ...]
    revision_direction_match: bool | None
    revision_direction_reasoning: str = ""


@dataclass(frozen=True)
class RuleProposal:
    kind: str
    text: str
    correction: str = ""
    rationale: str = ""
    duplicate_of: str = ""


@dataclass(frozen=True)
class RuleProposalSet:
    proposals: tuple[RuleProposal, ...]
    source_record_id: str
