from __future__ import annotations

from collections.abc import Iterable
import re

from decision_agent.engines import AgreementJudgment, CoreIssueJudgment
from decision_agent.models import (
    ArtifactReview,
    ArtifactReviewRequest,
    DecisionProfile,
    DecisionRecord,
    PatternEntry,
    PreferenceRule,
    ReviewIssue,
    UserFeedback,
)

ASCII_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")
SIGNAL_TOKEN_RE = re.compile(r"\w+")
MIN_USEFUL_ARTIFACT_LENGTH = 120
HISTORY_MATCH_LIMIT = 3
MIN_NON_ASCII_CHAR_NGRAM_LENGTH = 4
MIN_CHAR_UNIT_OVERLAP = 3


class HeuristicReviewEngine:
    name = "heuristic"

    def review(
        self,
        request: ArtifactReviewRequest,
        profile: DecisionProfile,
        records: tuple[DecisionRecord, ...],
    ) -> ArtifactReview:
        text = _review_text(request)
        issues: list[ReviewIssue] = []
        relevant_records = _relevant_records(request, records)

        if len(request.artifact.strip()) < MIN_USEFUL_ARTIFACT_LENGTH:
            issues.append(
                ReviewIssue(
                    severity="medium",
                    reason="artifact is too short to evaluate the user's judgment criteria reliably",
                    suggestion="add enough outline, script, or draft detail before asking for a final judgment",
                )
            )

        for mistake in profile.known_mistakes:
            if text_similarity(mistake.pattern, text) >= 0.2:
                issues.append(
                    ReviewIssue(
                        severity="high",
                        reason=f"resembles known mistake: {mistake.pattern}",
                        suggestion=mistake.correction,
                    )
                )

        for record in relevant_records[:HISTORY_MATCH_LIMIT]:
            rejected_before = record.user_feedback.verdict in {"revise", "reject"}
            similar_artifact = text_similarity(record.request.artifact, text) >= 0.2
            if rejected_before and similar_artifact:
                issues.append(
                    ReviewIssue(
                        severity="medium",
                        reason=f"similar past artifact was judged {record.user_feedback.verdict}: {record.user_feedback.notes}",
                        suggestion="compare against the past feedback before accepting this artifact",
                    )
                )

        active_negative_patterns = tuple(item for item in profile.negative_patterns if item.applies_to(request.task_type))
        negative_matches = _matched_items(active_negative_patterns, text)
        for pattern in negative_matches:
            issues.append(
                ReviewIssue(
                    severity="high",
                    reason=f"matches a negative pattern: {pattern.text}",
                    suggestion="revise the artifact so this pattern is removed or explicitly justified",
                    violated_rule_id=pattern.id,
                )
            )

        active_preference_rules = tuple(item for item in profile.preference_rules if item.applies_to(request.task_type))
        matched_rules = _matched_items(active_preference_rules, text)
        missing_rules = [rule for rule in active_preference_rules if rule not in matched_rules]
        for rule in missing_rules[:2]:
            issues.append(
                ReviewIssue(
                    severity="medium",
                    reason=f"does not clearly satisfy preference rule: {rule.text}",
                    suggestion="revise the artifact to make this preference visible in the structure or wording",
                    violated_rule_id=rule.id,
                )
            )

        active_positive_examples = tuple(item for item in profile.positive_examples if item.applies_to(request.task_type))
        positive_matches = _matched_items(active_positive_examples, text)
        if len([issue for issue in issues if issue.severity == "high"]) >= 2:
            verdict = "reject"
        elif issues:
            verdict = "revise"
        else:
            verdict = "accept"

        return ArtifactReview(
            verdict=verdict,
            confidence=_review_confidence(issues, matched_rules, positive_matches),
            summary=_review_summary(verdict, issues, matched_rules, positive_matches),
            issues=tuple(issues),
            revision_instruction=_revision_instruction(verdict, issues),
            learned_signals=(
                *(f"checked preference rule: {rule.id} {rule.text}" for rule in matched_rules[:3]),
                *(f"used past record: {record.id}" for record in relevant_records[:2] if record.id),
            ),
            engine=self.name,
        )


class HeuristicAgreementJudge:
    name = "heuristic"

    def judge(self, expected: UserFeedback, review: ArtifactReview) -> AgreementJudgment:
        review_signal_text = review_signal_text_for(review)
        core_issues = tuple(
            CoreIssueJudgment(
                issue=issue,
                noticed=text_matches_signal(issue, review_signal_text),
                evidence=_matched_evidence(issue, review_signal_text),
            )
            for issue in expected.core_issues
        )

        revision_direction_match = None
        revision_reasoning = ""
        if expected.revision_direction:
            revision_direction_match = text_matches_signal(expected.revision_direction, review.revision_instruction)
            revision_reasoning = _matched_evidence(expected.revision_direction, review.revision_instruction)

        return AgreementJudgment(
            core_issues=core_issues,
            revision_direction_match=revision_direction_match,
            revision_direction_reasoning=revision_reasoning,
        )


def review_signal_text_for(review: ArtifactReview) -> str:
    issue_text = " ".join(f"{issue.reason} {issue.suggestion}" for issue in review.issues)
    return f"{review.summary} {review.revision_instruction} {issue_text} {' '.join(review.learned_signals)}".lower()


def text_matches_signal(text: str, signal_text: str) -> bool:
    candidate = _normalize(text)
    if not candidate:
        return False
    normalized_signal = _normalize(signal_text)
    return candidate in normalized_signal or text_similarity(candidate, normalized_signal) >= 0.25


def text_similarity(left: str, right: str) -> float:
    left_normalized = _normalize(left)
    right_normalized = _normalize(right)
    if not left_normalized or not right_normalized:
        return 0.0
    if left_normalized in right_normalized:
        return 1.0

    return max(
        _token_containment(left_normalized, right_normalized),
        _char_ngram_containment(left_normalized, right_normalized),
    )


def _review_text(request: ArtifactReviewRequest) -> str:
    context = " ".join(request.context.values())
    return _normalize(f"{request.task_type} {request.intent} {context} {request.artifact}")


def _matched_items(items: tuple[PreferenceRule | PatternEntry, ...], text: str) -> list[PreferenceRule | PatternEntry]:
    return [item for item in items if text_matches_signal(item.text, text) or text_similarity(item.text, text) >= 0.34]


def _token_containment(left: str, right: str) -> float:
    left_tokens = _ascii_tokens(left)
    right_tokens = _ascii_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens)


def _ascii_tokens(text: str) -> set[str]:
    return {token.lower() for token in ASCII_TOKEN_RE.findall(text) if len(token) > 2}


def _char_ngram_containment(left: str, right: str) -> float:
    if not (_contains_non_ascii(left) or _contains_non_ascii(right)):
        return 0.0
    if len(re.sub(r"\s+", "", left)) < MIN_NON_ASCII_CHAR_NGRAM_LENGTH:
        return 0.0
    left_units = _char_units(left)
    right_units = _char_units(right)
    if not left_units or not right_units:
        return 0.0
    overlap = left_units & right_units
    if len(overlap) < MIN_CHAR_UNIT_OVERLAP:
        return 0.0
    return len(overlap) / len(left_units)


def _char_units(text: str) -> set[str]:
    compact = re.sub(r"\s+", "", text.lower())
    if not compact:
        return set()

    units: set[str] = set()
    if _contains_non_ascii(compact):
        units.update(char for char in compact if _is_meaningful_char(char))
    for size in (2, 3):
        if len(compact) >= size:
            units.update(compact[index : index + size] for index in range(len(compact) - size + 1))
    return units


def _contains_non_ascii(text: str) -> bool:
    return any(ord(char) > 127 for char in text)


def _is_meaningful_char(char: str) -> bool:
    return char.isalnum() and char not in {"の", "に", "を", "が", "は", "で", "と", "へ", "な", "る", "す"}


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


def _relevant_records(
    request: ArtifactReviewRequest,
    records: tuple[DecisionRecord, ...],
) -> list[DecisionRecord]:
    same_task = [record for record in records if record.request.task_type == request.task_type]
    return sorted(
        same_task,
        key=lambda record: (
            text_similarity(request.artifact, record.request.artifact),
            text_similarity(request.intent, record.request.intent),
            record.created_at,
        ),
        reverse=True,
    )


def _matched_evidence(pattern: str, text: str) -> str:
    normalized_pattern = _normalize(pattern)
    normalized_text = _normalize(text)
    if normalized_pattern and normalized_pattern in normalized_text:
        return pattern

    pattern_tokens = _signal_tokens(normalized_pattern)
    text_tokens = _signal_tokens(normalized_text)
    overlap = tuple(token for token in pattern_tokens if token in text_tokens)
    if overlap:
        return " ".join(overlap[:8])
    return _char_evidence(normalized_pattern, normalized_text)


def _signal_tokens(text: str) -> tuple[str, ...]:
    return tuple(token.lower() for token in SIGNAL_TOKEN_RE.findall(text) if len(token) > 2)


def _char_evidence(pattern: str, text: str) -> str:
    if text_similarity(pattern, text) < 0.25:
        return ""
    text_units = _char_units(text)
    overlap = [unit for unit in _ordered_char_units(pattern) if unit in text_units]
    return " ".join(overlap[:8])


def _ordered_char_units(text: str) -> tuple[str, ...]:
    compact = re.sub(r"\s+", "", text.lower())
    values: list[str] = []
    if _contains_non_ascii(compact):
        values.extend(char for char in compact if _is_meaningful_char(char))
    for size in (2, 3):
        if len(compact) >= size:
            values.extend(compact[index : index + size] for index in range(len(compact) - size + 1))

    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            deduped.append(value)
            seen.add(value)
    return tuple(deduped)


def _normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _append_unique(current: tuple[str, ...], additions: tuple[str, ...]) -> tuple[str, ...]:
    values = list(current)
    seen = set(current)
    for item in additions:
        if item and item not in seen:
            values.append(item)
            seen.add(item)
    return tuple(values)
