from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
from typing import TypeVar

from decision_agent.agent import FLAG_CONTRADICTS_ESTABLISHED_RULE, RECURRENCE_THRESHOLD, DecisionAgent
from decision_agent.engines import ReviewEngine
from decision_agent.engines.heuristic import HeuristicAgreementJudge, HeuristicReviewEngine
from decision_agent.engines.llm import LLMEngineError, LLMReviewEngine
from decision_agent.models import DecisionProfile, DecisionRecord, KnownMistake, PatternEntry, PreferenceRule
from decision_agent.storage import (
    append_decision_record,
    load_decision_records,
    load_evaluation_cases,
    load_feedback,
    load_legacy_profile_decision_records,
    load_profile,
    load_request,
    load_review,
    load_review_request,
    save_profile,
)


SUPPORTED_ENGINES = {"heuristic", "llm"}


def main(argv: list[str] | None = None) -> int:
    try:
        return _main(argv)
    except LLMEngineError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="decision-agent")
    subcommands = parser.add_subparsers(dest="command", required=True)

    decide_parser = subcommands.add_parser("decide", help="Rank alternatives for a request.")
    decide_parser.add_argument("profile", type=Path)
    decide_parser.add_argument("request", type=Path)

    train_parser = subcommands.add_parser("train", help="Update criteria weights from examples.")
    train_parser.add_argument("profile", type=Path)
    train_parser.add_argument("--output", "-o", type=Path, required=True)
    train_parser.add_argument("--learning-rate", type=float, default=0.2)

    review_parser = subcommands.add_parser("review", help="Review a subjective artifact.")
    review_parser.add_argument("profile", type=Path)
    review_parser.add_argument("request", type=Path)
    review_parser.add_argument("--records", type=Path, help="Read past decision records from JSONL.")
    _add_engine_arguments(review_parser)

    learn_parser = subcommands.add_parser("learn", help="Record user feedback from a review.")
    learn_parser.add_argument("profile", type=Path)
    learn_parser.add_argument("request", type=Path)
    learn_parser.add_argument("review", type=Path)
    learn_parser.add_argument("feedback", type=Path)
    learn_parser.add_argument("--output", "-o", type=Path, required=True)
    learn_parser.add_argument("--records", type=Path, help="Append the learned decision record to JSONL.")
    _add_engine_arguments(learn_parser)

    iterate_parser = subcommands.add_parser("iterate", help="Review, learn from feedback, and append history.")
    iterate_parser.add_argument("profile", type=Path)
    iterate_parser.add_argument("request", type=Path)
    iterate_parser.add_argument("--feedback", type=Path, required=True)
    iterate_parser.add_argument("--records", type=Path, required=True)
    iterate_parser.add_argument("--output", "-o", type=Path, required=True)
    _add_engine_arguments(iterate_parser)

    evaluate_parser = subcommands.add_parser("evaluate", help="Compare agent reviews against user judgments.")
    evaluate_parser.add_argument("profile", type=Path)
    evaluate_parser.add_argument("cases", type=Path)
    evaluate_parser.add_argument("--records", type=Path, help="Read past decision records from JSONL.")
    _add_engine_arguments(evaluate_parser)

    rules_parser = subcommands.add_parser("rules", help="List or update profile rules.")
    rules_subcommands = rules_parser.add_subparsers(dest="rules_command", required=True)

    rules_list_parser = rules_subcommands.add_parser("list", help="List profile rules and patterns.")
    rules_list_parser.add_argument("profile", type=Path)
    rules_list_parser.add_argument("--status", choices=("active", "candidate", "retired"))
    rules_list_parser.add_argument("--json", action="store_true", help="Print rules as JSON.")

    for command in ("approve", "reject", "retire"):
        command_parser = rules_subcommands.add_parser(command, help=f"{command} one profile rule.")
        command_parser.add_argument("profile", type=Path)
        command_parser.add_argument("rule_id")
        command_parser.add_argument("--output", "-o", type=Path)

    migrate_parser = subcommands.add_parser("migrate-history", help="Move legacy embedded profile records to JSONL.")
    migrate_parser.add_argument("profile", type=Path)
    migrate_parser.add_argument("--records", type=Path, required=True)
    migrate_parser.add_argument("--output", "-o", type=Path)

    args = parser.parse_args(argv)

    if args.command == "decide":
        profile = load_profile(args.profile)
        context, alternatives = load_request(args.request)
        result = DecisionAgent(profile).decide(context, alternatives)
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
        return 0

    if args.command == "train":
        profile = load_profile(args.profile)
        trained = DecisionAgent(profile).train(learning_rate=args.learning_rate)
        save_profile(trained, args.output)
        return 0

    if args.command == "review":
        profile = load_profile(args.profile)
        request = load_review_request(args.request)
        records = load_decision_records(args.records) if args.records else None
        agent = _agent(profile, args, parser)
        review = agent.review(request, history_records=records)
        print(json.dumps(review.to_dict(), indent=2, ensure_ascii=False))
        return 0

    if args.command == "learn":
        # learn() only records a pre-computed review + feedback; it never
        # calls a ReviewEngine. --engine is validated here (so a typo'd
        # value still errors) but intentionally not passed to _agent(),
        # which would construct an engine (and, for --engine llm, hit the
        # gateway) that this command never uses.
        _engine_name(args, parser)
        profile = load_profile(args.profile)
        request = load_review_request(args.request)
        review = load_review(args.review)
        feedback = load_feedback(args.feedback)
        learned = DecisionAgent(profile).learn(request, review, feedback)
        if args.records:
            append_decision_record(args.records, learned.decision_records[-1])
        save_profile(learned, args.output)
        print(
            json.dumps(
                _learning_summary(learned, learned.decision_records[-1]),
                indent=2,
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 0

    if args.command == "iterate":
        profile = load_profile(args.profile)
        request = load_review_request(args.request)
        feedback = load_feedback(args.feedback)
        records = load_decision_records(args.records)
        agent = _agent(profile, args, parser)
        review = agent.review(request, history_records=records)
        learned = agent.learn(request, review, feedback)
        append_decision_record(args.records, learned.decision_records[-1])
        save_profile(learned, args.output)
        print(
            json.dumps(
                {
                    "review": review.to_dict(),
                    "record": learned.decision_records[-1].to_dict(),
                    "profile": str(args.output),
                    "records": str(args.records),
                    "learning": _learning_summary(learned, learned.decision_records[-1]),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    if args.command == "evaluate":
        profile = load_profile(args.profile)
        cases = load_evaluation_cases(args.cases)
        if not cases:
            parser.error(f"no valid evaluation cases found in: {args.cases}")
        records = load_decision_records(args.records) if args.records else None
        report = _agent(profile, args, parser).evaluate(cases, history_records=records)
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
        return 0

    if args.command == "rules":
        profile = load_profile(args.profile)
        if args.rules_command == "list":
            rules = _rule_rows(profile, status=args.status)
            if args.json:
                print(json.dumps(rules, indent=2, ensure_ascii=False))
            else:
                for item in rules:
                    print(
                        "\t".join(
                            str(item[key])
                            for key in (
                                "id",
                                "kind",
                                "status",
                                "source",
                                "hit_count",
                                "miss_count",
                                "staleness",
                                "flagged_reason",
                                "text",
                            )
                        )
                    )
            return 0

        updated = _update_rule(profile, args.rule_id, args.rules_command, parser)
        save_profile(updated, args.output or args.profile)
        return 0

    if args.command == "migrate-history":
        profile = load_profile(args.profile)
        for record in load_legacy_profile_decision_records(args.profile):
            append_decision_record(args.records, record)
        save_profile(replace(profile, decision_records=()), args.output or args.profile)
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2


def _add_engine_arguments(command_parser: argparse.ArgumentParser) -> None:
    command_parser.add_argument(
        "--engine",
        choices=("heuristic", "llm"),
        help="Review engine to use. Defaults to DECISION_AGENT_ENGINE or heuristic. "
        "The llm engine delegates to local-agent-gateway; provider and model "
        "selection are gateway-side concerns.",
    )


def _agent(profile: DecisionProfile, args: argparse.Namespace, parser: argparse.ArgumentParser) -> DecisionAgent:
    engine = _engine_name(args, parser)
    if engine == "llm":
        review_engine: ReviewEngine = LLMReviewEngine()
    else:
        review_engine = HeuristicReviewEngine()
    return DecisionAgent(
        profile,
        review_engine=review_engine,
        agreement_judge=HeuristicAgreementJudge(),
    )


def _engine_name(args: argparse.Namespace, parser: argparse.ArgumentParser) -> str:
    engine = args.engine or os.environ.get("DECISION_AGENT_ENGINE", "heuristic")
    if engine not in SUPPORTED_ENGINES:
        parser.error(f"unsupported engine: {engine!r} (supported: {', '.join(sorted(SUPPORTED_ENGINES))})")
    return engine


def _flag_resolution_effect(entry: PreferenceRule | PatternEntry | KnownMistake) -> str:
    """Spell out what approve/reject concretely do for this flagged entry.

    For an ordinary flagged candidate, approve promotes the new thing and
    reject discards it -- the usual meaning. For a contradiction flagged
    against an already-established (active) entry, that mapping inverts:
    approve keeps the established rule and discards the new conflicting
    signal, reject accepts the new signal and overwrites the established
    one. Spelling this out avoids requiring the user to have internalized
    the inversion before running the command.
    """
    if entry.status == "active":
        return "approve = keep established (discard new signal); reject = accept new signal (overwrite established)"
    return "approve = promote this candidate; reject = discard this candidate"


def _learning_summary(profile: DecisionProfile, record: DecisionRecord) -> dict[str, object]:
    """Report what this learn() call did to candidate/active rule status,
    and which entries anywhere in the profile still await a confirmation
    decision. Per Philosophy, a rule taught in one learn() call may stay a
    candidate that does not yet apply to reviews -- this surfaces that
    explicitly so it reads as expected behavior, not a silent regression.
    """
    touched: list[dict[str, object]] = []
    needs_confirmation: list[dict[str, object]] = []
    for kind, entries in (
        *_profile_rule_groups(profile),
        ("known_mistake", profile.known_mistakes),
    ):
        for entry in entries:
            if isinstance(entry, KnownMistake):
                identifier, text = entry.pattern, entry.pattern
            else:
                identifier, text = entry.id, entry.text

            if entry.flagged_reason:
                needs_confirmation.append(
                    {
                        "kind": kind,
                        "id": identifier,
                        "flagged_reason": entry.flagged_reason,
                        "text": text,
                        "resolve_with": f"rules approve|reject {identifier}",
                        "effect": _flag_resolution_effect(entry),
                    }
                )

            if record.id not in entry.source_record_ids:
                continue
            distinct_records = len(set(entry.source_record_ids))
            remaining = max(0, RECURRENCE_THRESHOLD - distinct_records)
            touched.append(
                {
                    "kind": kind,
                    "id": identifier,
                    "status": entry.status,
                    "distinct_record_count": distinct_records,
                    "records_needed_to_activate": remaining,
                    "text": text,
                }
            )

    result: dict[str, object] = {"touched_rules": touched, "needs_confirmation": needs_confirmation}
    if record.flagged_reason:
        result["this_record_flagged_reason"] = record.flagged_reason
    return result


def _rule_rows(profile: DecisionProfile, *, status: str | None = None) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for kind, entries in _profile_rule_groups(profile):
        for entry in entries:
            if status and entry.status != status:
                continue
            rows.append(
                {
                    "id": entry.id,
                    "kind": kind,
                    "status": entry.status,
                    "source": entry.source,
                    "hit_count": entry.hit_count,
                    "miss_count": entry.miss_count,
                    "text": entry.text,
                    "staleness": _staleness_flag(entry),
                    "flagged_reason": entry.flagged_reason,
                }
            )
    return rows


def _staleness_flag(entry: PreferenceRule | PatternEntry) -> str:
    """Advisory-only staleness signal, never used to auto-retire a rule.

    Reuses RECURRENCE_THRESHOLD rather than introducing a new numeric
    constant, per Philosophy: staleness should be flagged for
    reconsideration, not silently acted on.
    """
    if entry.status != "active":
        return ""
    if entry.miss_count > entry.hit_count and entry.miss_count >= RECURRENCE_THRESHOLD:
        return "consider reviewing: repeatedly contradicted by user feedback"
    if not entry.last_used_at and entry.hit_count == 0 and entry.miss_count == 0:
        # hit/miss only wire up when this rule caused a ReviewIssue (see
        # _apply_rule_usage); "unexercised" means it has never surfaced an
        # issue, not literally never consulted during a review.
        return "unexercised: has not caused a review issue yet"
    return ""


def _update_rule(
    profile: DecisionProfile,
    rule_id: str,
    action: str,
    parser: argparse.ArgumentParser,
) -> DecisionProfile:
    updated = False

    preference_rules: list[PreferenceRule] = []
    for entry in profile.preference_rules:
        replacement, changed = _update_entry(entry, rule_id, action, parser)
        updated = updated or changed
        if replacement is not None:
            preference_rules.append(replacement)

    negative_patterns: list[PatternEntry] = []
    for entry in profile.negative_patterns:
        replacement, changed = _update_entry(entry, rule_id, action, parser)
        updated = updated or changed
        if replacement is not None:
            negative_patterns.append(replacement)

    positive_examples: list[PatternEntry] = []
    for entry in profile.positive_examples:
        replacement, changed = _update_entry(entry, rule_id, action, parser)
        updated = updated or changed
        if replacement is not None:
            positive_examples.append(replacement)

    known_mistakes: list[KnownMistake] = []
    for mistake in profile.known_mistakes:
        replacement, changed = _update_known_mistake(mistake, rule_id, action, parser)
        updated = updated or changed
        if replacement is not None:
            known_mistakes.append(replacement)

    if not updated:
        parser.error(f"rule not found: {rule_id}")

    return replace(
        profile,
        preference_rules=tuple(preference_rules),
        negative_patterns=tuple(negative_patterns),
        positive_examples=tuple(positive_examples),
        known_mistakes=tuple(known_mistakes),
    )


def _update_known_mistake(
    mistake: KnownMistake,
    rule_id: str,
    action: str,
    parser: argparse.ArgumentParser,
) -> tuple[KnownMistake | None, bool]:
    # KnownMistake has no stable id field; `rules list`/`_learning_summary`
    # already use `pattern` as its identifier, so resolution matches on it too.
    if mistake.pattern != rule_id:
        return mistake, False

    if mistake.flagged_reason == FLAG_CONTRADICTS_ESTABLISHED_RULE and mistake.status == "active":
        # A flagged, already-established mistake: approve keeps the
        # established correction/verdict and discards the new conflicting
        # signal; reject accepts the new signal and overwrites the
        # established correction/verdict. Either way the flag clears.
        if action == "approve":
            return replace(mistake, flagged_reason="", pending_correction="", pending_corrected_verdict=""), True
        if action == "reject":
            return (
                replace(
                    mistake,
                    correction=mistake.pending_correction or mistake.correction,
                    corrected_verdict=mistake.pending_corrected_verdict or mistake.corrected_verdict,
                    flagged_reason="",
                    pending_correction="",
                    pending_corrected_verdict="",
                ),
                True,
            )
        if action == "retire":
            return replace(mistake, status="retired", flagged_reason=""), True
        parser.error(f"unknown rules command: {action}")
        raise AssertionError("unreachable")

    if action == "approve":
        if mistake.status != "candidate":
            parser.error(f"only candidate rules can be approved: {rule_id}")
        return replace(mistake, status="active", flagged_reason=""), True
    if action == "reject":
        if mistake.status != "candidate":
            parser.error(f"only candidate rules can be rejected: {rule_id}")
        return None, True
    if action == "retire":
        return replace(mistake, status="retired", flagged_reason=""), True
    parser.error(f"unknown rules command: {action}")
    raise AssertionError("unreachable")


RuleEntry = TypeVar("RuleEntry", PreferenceRule, PatternEntry)


def _update_entry(
    entry: RuleEntry,
    rule_id: str,
    action: str,
    parser: argparse.ArgumentParser,
) -> tuple[RuleEntry | None, bool]:
    if entry.id != rule_id:
        return entry, False

    if action == "approve":
        if entry.status != "candidate":
            parser.error(f"only candidate rules can be approved: {rule_id}")
        # Approving a flagged candidate is an explicit override of the
        # conflict it was flagged for; clear the flag once resolved.
        return replace(entry, status="active", flagged_reason=""), True
    if action == "reject":
        if entry.status != "candidate":
            parser.error(f"only candidate rules can be rejected: {rule_id}")
        return None, True
    if action == "retire":
        return replace(entry, status="retired", flagged_reason=""), True
    parser.error(f"unknown rules command: {action}")
    raise AssertionError("unreachable")


def _profile_rule_groups(profile: DecisionProfile):
    return (
        ("preference_rule", profile.preference_rules),
        ("negative_pattern", profile.negative_patterns),
        ("positive_example", profile.positive_examples),
    )


if __name__ == "__main__":
    raise SystemExit(main())
