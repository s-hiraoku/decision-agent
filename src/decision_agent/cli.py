from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from decision_agent.agent import DecisionAgent
from decision_agent.engines.heuristic import HeuristicAgreementJudge, HeuristicReviewEngine
from decision_agent.storage import (
    append_decision_record,
    load_decision_records,
    load_evaluation_cases,
    load_feedback,
    load_profile,
    load_request,
    load_review,
    load_review_request,
    save_profile,
)


SUPPORTED_ENGINES = {"heuristic"}


def main(argv: list[str] | None = None) -> int:
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
        _engine_name(args, parser)
        profile = load_profile(args.profile)
        request = load_review_request(args.request)
        review = load_review(args.review)
        feedback = load_feedback(args.feedback)
        learned = DecisionAgent(profile).learn(request, review, feedback)
        if args.records:
            append_decision_record(args.records, learned.decision_records[-1])
        save_profile(learned, args.output)
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

    parser.error(f"unknown command: {args.command}")
    return 2


def _add_engine_arguments(command_parser: argparse.ArgumentParser) -> None:
    command_parser.add_argument(
        "--engine",
        choices=("heuristic", "llm"),
        help="Review engine to use. Defaults to DECISION_AGENT_ENGINE or heuristic.",
    )
    command_parser.add_argument("--model", help="LLM model name. Reserved for the llm engine.")


def _agent(profile, args: argparse.Namespace, parser: argparse.ArgumentParser) -> DecisionAgent:
    _engine_name(args, parser)
    return DecisionAgent(
        profile,
        review_engine=HeuristicReviewEngine(),
        agreement_judge=HeuristicAgreementJudge(),
    )


def _engine_name(args: argparse.Namespace, parser: argparse.ArgumentParser) -> str:
    engine = args.engine or os.environ.get("DECISION_AGENT_ENGINE", "heuristic")
    if engine not in SUPPORTED_ENGINES:
        parser.error("only the heuristic engine is implemented; llm support is planned in docs/detailed-design.md")
    if args.model and engine == "heuristic":
        parser.error("--model is only valid with --engine llm")
    return engine


if __name__ == "__main__":
    raise SystemExit(main())
