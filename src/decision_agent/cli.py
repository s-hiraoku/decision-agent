from __future__ import annotations

import argparse
import json
from pathlib import Path

from decision_agent.agent import DecisionAgent
from decision_agent.storage import (
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

    learn_parser = subcommands.add_parser("learn", help="Record user feedback from a review.")
    learn_parser.add_argument("profile", type=Path)
    learn_parser.add_argument("request", type=Path)
    learn_parser.add_argument("review", type=Path)
    learn_parser.add_argument("feedback", type=Path)
    learn_parser.add_argument("--output", "-o", type=Path, required=True)

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
        review = DecisionAgent(profile).review(request)
        print(json.dumps(review.to_dict(), indent=2, ensure_ascii=False))
        return 0

    if args.command == "learn":
        profile = load_profile(args.profile)
        request = load_review_request(args.request)
        review = load_review(args.review)
        feedback = load_feedback(args.feedback)
        learned = DecisionAgent(profile).learn(request, review, feedback)
        save_profile(learned, args.output)
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
