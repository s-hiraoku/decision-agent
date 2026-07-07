from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import replace
from datetime import UTC, datetime
import math
import re
import uuid

from decision_agent.engines import AgreementJudge, ReviewEngine
from decision_agent.engines.heuristic import HeuristicAgreementJudge, HeuristicReviewEngine
from decision_agent.models import (
    Alternative,
    ArtifactReview,
    ArtifactReviewRequest,
    DecisionExample,
    DecisionProfile,
    DecisionRecord,
    DecisionResult,
    EvaluationCase,
    EvaluationCaseResult,
    EvaluationReport,
    KnownMistake,
    PatternEntry,
    PreferenceRule,
    SUPPORTED_VERDICTS,
    UserFeedback,
)

ATTRIBUTE_SCALE = 10.0
MEMORY_WEIGHT = 0.25
TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


class DecisionAgent:
    def __init__(
        self,
        profile: DecisionProfile,
        *,
        review_engine: ReviewEngine | None = None,
        agreement_judge: AgreementJudge | None = None,
    ):
        self.profile = profile
        self.review_engine = review_engine or HeuristicReviewEngine()
        self.agreement_judge = agreement_judge or HeuristicAgreementJudge()

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

    def review(
        self,
        request: ArtifactReviewRequest,
        history_records: tuple[DecisionRecord, ...] | None = None,
    ) -> ArtifactReview:
        records = self.profile.decision_records if history_records is None else history_records
        review = self.review_engine.review(request, self.profile, records)
        if review.verdict not in SUPPORTED_VERDICTS:
            raise ValueError(f"unsupported verdict: {review.verdict}")
        engine_name = review.engine or getattr(self.review_engine, "name", "")
        return replace(review, confidence=max(0.0, min(1.0, review.confidence)), engine=engine_name)

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
            id=_record_id(request),
            created_at=datetime.now(UTC).isoformat(),
        )

        return replace(
            self.profile,
            preference_rules=_append_preference_rules(self.profile.preference_rules, user_feedback.preference_rules),
            negative_patterns=_append_pattern_entries(
                self.profile.negative_patterns,
                user_feedback.negative_patterns,
                kind="negative_pattern",
            ),
            positive_examples=_append_pattern_entries(
                self.profile.positive_examples,
                user_feedback.positive_examples,
                kind="positive_example",
            ),
            known_mistakes=_update_known_mistakes(self.profile.known_mistakes, agent_review, user_feedback),
            decision_records=(*self.profile.decision_records, record),
        )

    def evaluate(
        self,
        cases: tuple[EvaluationCase, ...],
        history_records: tuple[DecisionRecord, ...] | None = None,
    ) -> EvaluationReport:
        records = self.profile.decision_records if history_records is None else history_records
        results: list[EvaluationCaseResult] = []

        for index, case in enumerate(cases, start=1):
            review = self.review(case.request, history_records=records)
            judgment = case.user_judgment
            agreement = self.agreement_judge.judge(judgment, review)
            missed_core_issues = tuple(result.issue for result in agreement.core_issues if not result.noticed)
            core_issue_agreement = None
            if judgment.core_issues:
                core_issue_agreement = len(missed_core_issues) == 0

            revision_direction_agreement = agreement.revision_direction_match

            suggested_updates = _evaluation_profile_updates(
                case,
                review,
                missed_core_issues,
                revision_direction_agreement,
            )
            results.append(
                EvaluationCaseResult(
                    id=case.id or f"case-{index}",
                    agent_verdict=review.verdict,
                    user_verdict=judgment.verdict,
                    verdict_agreement=review.verdict == judgment.verdict,
                    core_issue_agreement=core_issue_agreement,
                    revision_direction_agreement=revision_direction_agreement,
                    missed_core_issues=missed_core_issues,
                    suggested_profile_updates=suggested_updates,
                )
            )

        return EvaluationReport(
            cases=len(cases),
            verdict_accuracy=_boolean_accuracy(result.verdict_agreement for result in results),
            core_issue_accuracy=_optional_boolean_accuracy(result.core_issue_agreement for result in results),
            revision_direction_accuracy=_optional_boolean_accuracy(
                result.revision_direction_agreement for result in results
            ),
            common_misses=_common_misses(results),
            suggested_profile_updates=_unique_suggestions(results),
            case_results=tuple(results),
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


def _feedback_delta(agent_review: ArtifactReview, user_feedback: UserFeedback) -> str:
    if agent_review.verdict == user_feedback.verdict:
        return "agent verdict matched user feedback"
    return f"agent predicted {agent_review.verdict}, user judged {user_feedback.verdict}: {user_feedback.notes}"


def _evaluation_profile_updates(
    case: EvaluationCase,
    review: ArtifactReview,
    missed_core_issues: tuple[str, ...],
    revision_direction_agreement: bool | None,
) -> tuple[str, ...]:
    updates: list[str] = []
    judgment = case.user_judgment

    if review.verdict != judgment.verdict:
        reason = judgment.notes or judgment.revision_direction or "user judgment differed from agent judgment"
        updates.append(f"for {case.request.task_type}, prefer {judgment.verdict} when: {reason}")

    for issue in missed_core_issues:
        updates.append(f"for {case.request.task_type}, check whether: {issue}")

    if judgment.revision_direction and revision_direction_agreement is False:
        updates.append(f"for {case.request.task_type}, prefer revision direction: {judgment.revision_direction}")

    return _append_unique((), tuple(updates))


def _boolean_accuracy(values: Iterable[bool]) -> float:
    items = list(values)
    if not items:
        return 0.0
    return round(sum(1 for item in items if item) / len(items), 4)


def _optional_boolean_accuracy(values: Iterable[bool | None]) -> float | None:
    items = [item for item in values if item is not None]
    if not items:
        return None
    return round(sum(1 for item in items if item) / len(items), 4)


def _common_misses(results: list[EvaluationCaseResult]) -> tuple[str, ...]:
    counter = Counter(issue for result in results for issue in result.missed_core_issues)
    return tuple(issue for issue, _count in counter.most_common(5))


def _unique_suggestions(results: list[EvaluationCaseResult]) -> tuple[str, ...]:
    suggestions: tuple[str, ...] = ()
    for result in results:
        suggestions = _append_unique(suggestions, result.suggested_profile_updates)
    return suggestions[:10]


def _record_id(request: ArtifactReviewRequest) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    intent_tokens = [token.lower() for token in TOKEN_RE.findall(request.intent) if len(token) > 2]
    intent = "-".join(sorted(intent_tokens)[:4]) or "review"
    return f"{timestamp}-{request.task_type}-{intent}-{uuid.uuid4().hex[:8]}"


def _update_known_mistakes(
    current: tuple[KnownMistake, ...],
    agent_review: ArtifactReview,
    user_feedback: UserFeedback,
) -> tuple[KnownMistake, ...]:
    if agent_review.verdict == user_feedback.verdict:
        return current

    pattern = user_feedback.notes.strip() or f"agent predicted {agent_review.verdict} when user judged {user_feedback.verdict}"
    correction = _mistake_correction(user_feedback)
    values: list[KnownMistake] = []
    updated = False

    for mistake in current:
        if mistake.pattern == pattern:
            values.append(replace(mistake, count=mistake.count + 1, correction=correction or mistake.correction))
            updated = True
        else:
            values.append(mistake)

    if not updated:
        values.append(KnownMistake(pattern=pattern, correction=correction, count=1))

    return tuple(values)


def _mistake_correction(user_feedback: UserFeedback) -> str:
    if user_feedback.preference_rules:
        return " ".join(user_feedback.preference_rules)
    if user_feedback.negative_patterns:
        return "avoid: " + "; ".join(user_feedback.negative_patterns)
    if user_feedback.notes:
        return user_feedback.notes
    return "ask for user feedback before accepting similar artifacts"


def _append_unique(current: tuple[str, ...], additions: tuple[str, ...]) -> tuple[str, ...]:
    values = list(current)
    seen = set(current)
    for item in additions:
        if item and item not in seen:
            values.append(item)
            seen.add(item)
    return tuple(values)


def _append_preference_rules(
    current: tuple[PreferenceRule, ...],
    additions: tuple[str, ...],
) -> tuple[PreferenceRule, ...]:
    values = list(current)
    seen = {item.id for item in current}
    for item in additions:
        entry = PreferenceRule.from_value({"text": item, "source": "feedback"})
        if entry.text and entry.id not in seen:
            values.append(entry)
            seen.add(entry.id)
    return tuple(values)


def _append_pattern_entries(
    current: tuple[PatternEntry, ...],
    additions: tuple[str, ...],
    *,
    kind: str,
) -> tuple[PatternEntry, ...]:
    values = list(current)
    seen = {item.id for item in current}
    for item in additions:
        entry = PatternEntry.from_value({"text": item, "source": "feedback"}, kind=kind)
        if entry.text and entry.id not in seen:
            values.append(entry)
            seen.add(entry.id)
    return tuple(values)
