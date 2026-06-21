from __future__ import annotations

import math
import re
from dataclasses import replace

from decision_agent.models import (
    Alternative,
    ArtifactReview,
    ArtifactReviewRequest,
    DecisionExample,
    DecisionProfile,
    DecisionRecord,
    DecisionResult,
    ReviewIssue,
    UserFeedback,
)

ATTRIBUTE_SCALE = 10.0
MEMORY_WEIGHT = 0.25
TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")
REVIEW_TOKEN_RE = re.compile(r"\w+")
MIN_USEFUL_ARTIFACT_LENGTH = 120


class DecisionAgent:
    def __init__(self, profile: DecisionProfile):
        self.profile = profile

    def decide(self, context: str, alternatives: list[Alternative]) -> DecisionResult:
        if not self.profile.criteria:
            raise ValueError("profile criteria must not be empty")
        if not alternatives:
            raise ValueError("alternatives must not be empty")

        scores: dict[str, float] = {}
        explanations: dict[str, list[str]] = {}

        for alternative in alternatives:
            criterion_score, criterion_lines = self._criterion_score(alternative)
            memory_score = self._memory_score(context, alternative)
            total = criterion_score + memory_score
            scores[alternative.name] = round(total, 4)
            explanations[alternative.name] = [
                *criterion_lines,
                f"memory={memory_score:.3f}",
                f"total={total:.3f}",
            ]

        recommended = max(scores.items(), key=lambda item: (item[1], item[0]))[0]
        return DecisionResult(recommended=recommended, scores=scores, explanations=explanations)

    def train(self, learning_rate: float = 0.2) -> DecisionProfile:
        if not self.profile.criteria:
            raise ValueError("profile criteria must not be empty")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        deltas = {criterion: 0.0 for criterion in self.profile.criteria}
        counts = {criterion: 0 for criterion in self.profile.criteria}

        for example in self.profile.examples:
            chosen = example.alternative_by_name(example.chosen)
            if chosen is None:
                continue
            rejected = self._rejected_alternatives(example)
            for alternative in rejected:
                for criterion in deltas:
                    chosen_value = self._attribute_value(chosen, criterion)
                    rejected_value = self._attribute_value(alternative, criterion)
                    deltas[criterion] += chosen_value - rejected_value
                    counts[criterion] += 1

        criteria = dict(self.profile.criteria)
        for criterion, total_delta in deltas.items():
            if counts[criterion] == 0:
                continue
            criteria[criterion] = max(0.0, criteria[criterion] + learning_rate * total_delta / counts[criterion])

        return replace(self.profile, criteria=_normalize_weights(criteria))

    def review(self, request: ArtifactReviewRequest) -> ArtifactReview:
        text = _review_text(request)
        issues: list[ReviewIssue] = []

        if len(request.artifact.strip()) < MIN_USEFUL_ARTIFACT_LENGTH:
            issues.append(
                ReviewIssue(
                    severity="medium",
                    reason="artifact is too short to evaluate the user's judgment criteria reliably",
                    suggestion="add enough outline, script, or draft detail before asking for a final judgment",
                )
            )

        negative_matches = _matched_items(self.profile.negative_patterns, text)
        for pattern in negative_matches:
            issues.append(
                ReviewIssue(
                    severity="high",
                    reason=f"matches a negative pattern: {pattern}",
                    suggestion="revise the artifact so this pattern is removed or explicitly justified",
                )
            )

        matched_rules = _matched_items(self.profile.preference_rules, text)
        missing_rules = [rule for rule in self.profile.preference_rules if rule not in matched_rules]
        for rule in missing_rules[:2]:
            issues.append(
                ReviewIssue(
                    severity="medium",
                    reason=f"does not clearly satisfy preference rule: {rule}",
                    suggestion="revise the artifact to make this preference visible in the structure or wording",
                )
            )

        positive_matches = _matched_items(self.profile.positive_examples, text)
        if len([issue for issue in issues if issue.severity == "high"]) >= 2:
            verdict = "reject"
        elif issues:
            verdict = "revise"
        else:
            verdict = "accept"

        confidence = _review_confidence(issues, matched_rules, positive_matches)
        summary = _review_summary(verdict, issues, matched_rules, positive_matches)
        revision_instruction = _revision_instruction(verdict, issues)
        learned_signals = tuple(f"checked preference rule: {rule}" for rule in matched_rules[:3])

        return ArtifactReview(
            verdict=verdict,
            confidence=confidence,
            summary=summary,
            issues=tuple(issues),
            revision_instruction=revision_instruction,
            learned_signals=learned_signals,
        )

    def learn(
        self,
        request: ArtifactReviewRequest,
        agent_review: ArtifactReview,
        user_feedback: UserFeedback,
    ) -> DecisionProfile:
        delta = _feedback_delta(agent_review, user_feedback)
        record = DecisionRecord(
            request=request,
            agent_review=agent_review,
            user_feedback=user_feedback,
            delta=delta,
        )

        return replace(
            self.profile,
            preference_rules=_append_unique(self.profile.preference_rules, user_feedback.preference_rules),
            negative_patterns=_append_unique(self.profile.negative_patterns, user_feedback.negative_patterns),
            positive_examples=_append_unique(self.profile.positive_examples, user_feedback.positive_examples),
            decision_records=(*self.profile.decision_records, record),
        )

    def _criterion_score(self, alternative: Alternative) -> tuple[float, list[str]]:
        weights = _normalize_weights(self.profile.criteria)
        score = 0.0
        explanations: list[str] = []

        for criterion, weight in sorted(weights.items()):
            value = self._attribute_value(alternative, criterion)
            contribution = weight * value
            score += contribution
            explanations.append(f"{criterion}={value:.2f}*{weight:.2f}->{contribution:.3f}")

        return score, explanations

    def _memory_score(self, context: str, alternative: Alternative) -> float:
        if not self.profile.examples:
            return 0.0

        score = 0.0
        count = 0

        for example in self.profile.examples:
            context_multiplier = 0.5 + 0.5 * _token_similarity(context, example.context)
            chosen = example.alternative_by_name(example.chosen)
            if chosen is not None:
                score += context_multiplier * _attribute_similarity(alternative, chosen)
                count += 1
            for rejected in self._rejected_alternatives(example):
                score -= context_multiplier * _attribute_similarity(alternative, rejected)
                count += 1

        if count == 0:
            return 0.0
        return MEMORY_WEIGHT * score / count

    def _rejected_alternatives(self, example: DecisionExample) -> list[Alternative]:
        if example.rejected:
            return [item for item in example.alternatives if item.name in set(example.rejected)]
        return [item for item in example.alternatives if item.name != example.chosen]

    def _attribute_value(self, alternative: Alternative, criterion: str) -> float:
        return max(0.0, min(1.0, alternative.attributes.get(criterion, 0.0) / ATTRIBUTE_SCALE))


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    clipped = {key: max(0.0, value) for key, value in weights.items()}
    total = sum(clipped.values())
    if total == 0:
        equal = 1.0 / len(clipped)
        return {key: equal for key in clipped}
    return {key: value / total for key, value in clipped.items()}


def _attribute_similarity(left: Alternative, right: Alternative) -> float:
    keys = set(left.attributes) | set(right.attributes)
    if not keys:
        return 0.0

    left_vector = [left.attributes.get(key, 0.0) / ATTRIBUTE_SCALE for key in sorted(keys)]
    right_vector = [right.attributes.get(key, 0.0) / ATTRIBUTE_SCALE for key in sorted(keys)]
    left_norm = math.sqrt(sum(value * value for value in left_vector))
    right_norm = math.sqrt(sum(value * value for value in right_vector))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return sum(a * b for a, b in zip(left_vector, right_vector)) / (left_norm * right_norm)


def _token_similarity(left: str, right: str) -> float:
    left_tokens = set(TOKEN_RE.findall(left.lower()))
    right_tokens = set(TOKEN_RE.findall(right.lower()))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _review_text(request: ArtifactReviewRequest) -> str:
    context = " ".join(request.context.values())
    return f"{request.task_type} {request.intent} {context} {request.artifact}".lower()


def _matched_items(items: tuple[str, ...], text: str) -> list[str]:
    return [item for item in items if _text_similarity(item, text) >= 0.34 or item.lower() in text]


def _text_similarity(left: str, right: str) -> float:
    left_tokens = _review_tokens(left)
    right_tokens = _review_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens)


def _review_tokens(text: str) -> set[str]:
    return {token.lower() for token in REVIEW_TOKEN_RE.findall(text) if len(token) > 2}


def _review_confidence(
    issues: list[ReviewIssue],
    matched_rules: list[str],
    positive_matches: list[str],
) -> float:
    score = 0.45 + 0.08 * len(matched_rules) + 0.05 * len(positive_matches) + 0.04 * len(issues)
    return round(max(0.2, min(0.9, score)), 2)


def _review_summary(
    verdict: str,
    issues: list[ReviewIssue],
    matched_rules: list[str],
    positive_matches: list[str],
) -> str:
    if verdict == "accept":
        return "artifact appears aligned with the stored user preference profile"
    if matched_rules or positive_matches:
        return "artifact has some alignment, but stored preferences indicate revisions are needed"
    if issues:
        return "artifact needs revision before it matches the stored user judgment profile"
    return "not enough preference evidence is available for a strong judgment"


def _revision_instruction(verdict: str, issues: list[ReviewIssue]) -> str:
    if verdict == "accept":
        return "keep the current direction"
    if not issues:
        return "ask the user for feedback and record the judgment delta"
    suggestions = _append_unique((), tuple(issue.suggestion for issue in issues))
    return " ".join(f"{suggestion}." for suggestion in suggestions[:3])


def _feedback_delta(agent_review: ArtifactReview, user_feedback: UserFeedback) -> str:
    if agent_review.verdict == user_feedback.verdict:
        return "agent verdict matched user feedback"
    return f"agent predicted {agent_review.verdict}, user judged {user_feedback.verdict}: {user_feedback.notes}"


def _append_unique(current: tuple[str, ...], additions: tuple[str, ...]) -> tuple[str, ...]:
    values = list(current)
    seen = set(current)
    for item in additions:
        if item and item not in seen:
            values.append(item)
            seen.add(item)
    return tuple(values)
