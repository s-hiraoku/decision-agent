from __future__ import annotations

import argparse
import json
from pathlib import Path

from decision_agent.agent import DecisionAgent
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

    learn_parser = subcommands.add_parser("learn", help="Record user feedback from a review.")
    learn_parser.add_argument("profile", type=Path)
    learn_parser.add_argument("request", type=Path)
    learn_parser.add_argument("review", type=Path)
    learn_parser.add_argument("feedback", type=Path)
    learn_parser.add_argument("--output", "-o", type=Path, required=True)
    learn_parser.add_argument("--records", type=Path, help="Append the learned decision record to JSONL.")

    iterate_parser = subcommands.add_parser("iterate", help="Review, learn from feedback, and append history.")
    iterate_parser.add_argument("profile", type=Path)
    iterate_parser.add_argument("request", type=Path)
    iterate_parser.add_argument("--feedback", type=Path, required=True)
    iterate_parser.add_argument("--records", type=Path, required=True)
    iterate_parser.add_argument("--output", "-o", type=Path, required=True)

    evaluate_parser = subcommands.add_parser("evaluate", help="Compare agent reviews against user judgments.")
    evaluate_parser.add_argument("profile", type=Path)
    evaluate_parser.add_argument("cases", type=Path)
    evaluate_parser.add_argument("--records", type=Path, help="Read past decision records from JSONL.")

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
        review = DecisionAgent(profile).review(request, history_records=records)
        print(json.dumps(review.to_dict(), indent=2, ensure_ascii=False))
        return 0

    if args.command == "learn":
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
        agent = DecisionAgent(profile)
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
        records = load_decision_records(args.records) if args.records else None
        report = DecisionAgent(profile).evaluate(cases, history_records=records)
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
