import json
import unittest
from tempfile import TemporaryDirectory

from decision_agent.agent import DecisionAgent
from decision_agent.models import (
    Alternative,
    ArtifactReviewRequest,
    ArtifactReview,
    DecisionExample,
    DecisionProfile,
    DecisionRecord,
    EvaluationCase,
    UserFeedback,
)
from decision_agent.storage import append_decision_record, load_decision_records, load_evaluation_cases


class DecisionAgentTest(unittest.TestCase):
    def test_decide_prefers_profile_weighted_alternative(self) -> None:
        profile = DecisionProfile(
            user_id="u1",
            criteria={"quality": 0.7, "cost": 0.3},
        )
        agent = DecisionAgent(profile)

        result = agent.decide(
            "pick a laptop",
            [
                Alternative("cheap", {"quality": 4, "cost": 10}),
                Alternative("durable", {"quality": 9, "cost": 5}),
            ],
        )

        self.assertEqual(result.recommended, "durable")
        self.assertGreater(result.scores["durable"], result.scores["cheap"])

    def test_train_increases_weights_for_chosen_advantages(self) -> None:
        profile = DecisionProfile(
            user_id="u1",
            criteria={"quality": 0.25, "cost": 0.75},
            examples=(
                DecisionExample(
                    context="pick a vendor",
                    alternatives=(
                        Alternative("low-cost", {"quality": 4, "cost": 10}),
                        Alternative("reliable", {"quality": 9, "cost": 5}),
                    ),
                    chosen="reliable",
                ),
            ),
        )

        trained = DecisionAgent(profile).train(learning_rate=0.5)

        self.assertGreater(trained.criteria["quality"], profile.criteria["quality"])
        self.assertLess(trained.criteria["cost"], profile.criteria["cost"])

    def test_memory_score_can_break_close_tie(self) -> None:
        profile = DecisionProfile(
            user_id="u1",
            criteria={"quality": 0.5, "speed": 0.5},
            examples=(
                DecisionExample(
                    context="urgent production support",
                    alternatives=(
                        Alternative("steady", {"quality": 8, "speed": 5}),
                        Alternative("rapid", {"quality": 6, "speed": 8}),
                    ),
                    chosen="steady",
                    rejected=("rapid",),
                ),
            ),
        )

        result = DecisionAgent(profile).decide(
            "urgent production support",
            [
                Alternative("balanced-a", {"quality": 8, "speed": 5}),
                Alternative("balanced-b", {"quality": 6, "speed": 7}),
            ],
        )

        self.assertEqual(result.recommended, "balanced-a")

    def test_review_flags_missing_preference_rule(self) -> None:
        profile = DecisionProfile(
            user_id="u1",
            criteria={},
            preference_rules=("put a concrete pain point before abstract concept explanation",),
            negative_patterns=("abstract explanation before concrete problem",),
        )
        request = ArtifactReviewRequest(
            task_type="blog_outline",
            intent="write about Decision Agent",
            artifact=(
                "This post defines the concept first. It then explains loop engineering and describes "
                "why preference profiles can make agents better over repeated iterations."
            ),
        )

        review = DecisionAgent(profile).review(request)

        self.assertEqual(review.verdict, "revise")
        self.assertTrue(review.issues)
        self.assertIn("revise", review.revision_instruction)

    def test_learn_records_feedback_and_updates_profile(self) -> None:
        profile = DecisionProfile(user_id="u1", criteria={})
        request = ArtifactReviewRequest(
            task_type="blog_outline",
            intent="write about Decision Agent",
            artifact="A short outline about Decision Agent.",
        )
        agent = DecisionAgent(profile)
        review = agent.review(request)
        feedback = UserFeedback(
            verdict="revise",
            notes="Start with user pain.",
            preference_rules=("start with a concrete failure case",),
            negative_patterns=("concept definition before user pain",),
        )

        learned = agent.learn(request, review, feedback)

        self.assertEqual(len(learned.decision_records), 1)
        self.assertIn("start with a concrete failure case", learned.preference_rules)
        self.assertIn("concept definition before user pain", learned.negative_patterns)

    def test_learn_promotes_verdict_delta_to_known_mistake(self) -> None:
        profile = DecisionProfile(user_id="u1", criteria={})
        request = ArtifactReviewRequest(
            task_type="blog_outline",
            intent="write about Decision Agent",
            artifact=(
                "A detailed outline about Decision Agent. It explains the context, the loop, "
                "the profile, the feedback path, and a concrete next step for readers who want "
                "to improve agent output review quality over time."
            ),
        )
        agent = DecisionAgent(profile)
        review = agent.review(request)
        feedback = UserFeedback(
            verdict="reject",
            notes="Problem framing is too weak.",
            preference_rules=("start with a concrete failure before the concept",),
        )

        learned = agent.learn(request, review, feedback)

        self.assertEqual(len(learned.known_mistakes), 1)
        self.assertEqual(learned.known_mistakes[0].pattern, "Problem framing is too weak.")
        self.assertIn("concrete failure", learned.known_mistakes[0].correction)

    def test_review_uses_past_records_for_same_task_type(self) -> None:
        profile = DecisionProfile(user_id="u1", criteria={})
        request = ArtifactReviewRequest(
            task_type="blog_outline",
            intent="write about Decision Agent",
            artifact=(
                "A detailed outline about Decision Agent. It explains the concept, profile, "
                "feedback, and loop mechanics before showing why the reader should care about "
                "judgment alignment in creative agent workflows."
            ),
        )
        agent = DecisionAgent(profile)
        review = agent.review(request)
        feedback = UserFeedback(verdict="revise", notes="The concrete pain arrives too late.")
        learned = agent.learn(request, review, feedback)

        next_review = DecisionAgent(profile).review(request, history_records=learned.decision_records)

        self.assertEqual(next_review.verdict, "revise")
        self.assertTrue(any("similar past artifact" in issue.reason for issue in next_review.issues))

    def test_review_honors_explicitly_empty_history_records(self) -> None:
        request = ArtifactReviewRequest(
            task_type="blog_outline",
            intent="write about Decision Agent",
            artifact=(
                "A detailed outline about Decision Agent. It explains the concept, profile, "
                "feedback, and loop mechanics before showing why the reader should care about "
                "judgment alignment in creative agent workflows."
            ),
        )
        record = DecisionRecord(
            request=request,
            agent_review=ArtifactReview(verdict="revise", confidence=0.5, summary="needs work"),
            user_feedback=UserFeedback(verdict="revise", notes="The concrete pain arrives too late."),
            delta="agent verdict matched user feedback",
        )
        profile = DecisionProfile(user_id="u1", criteria={}, decision_records=(record,))

        review = DecisionAgent(profile).review(request, history_records=())

        self.assertEqual(review.verdict, "accept")
        self.assertFalse(any("similar past artifact" in issue.reason for issue in review.issues))

    def test_history_prefers_artifact_similarity_before_limit(self) -> None:
        request = ArtifactReviewRequest(
            task_type="blog_outline",
            intent="decision agent launch post",
            artifact=(
                "A concrete failure story opens the Decision Agent post. The artifact shows an "
                "agent creating a plausible outline, the user rejecting it, and the loop learning "
                "from that judgment delta."
            ),
        )
        unrelated_records = tuple(
            DecisionRecord(
                request=ArtifactReviewRequest(
                    task_type="blog_outline",
                    intent="decision agent launch post",
                    artifact=f"Unrelated draft {index} about packaging, release notes, setup, and repository hygiene.",
                ),
                agent_review=ArtifactReview(verdict="revise", confidence=0.5, summary="needs work"),
                user_feedback=UserFeedback(verdict="revise", notes=f"unrelated {index}"),
                delta="agent verdict matched user feedback",
                id=f"unrelated-{index}",
            )
            for index in range(3)
        )
        similar_record = DecisionRecord(
            request=request,
            agent_review=ArtifactReview(verdict="revise", confidence=0.5, summary="needs work"),
            user_feedback=UserFeedback(verdict="revise", notes="important similar artifact feedback"),
            delta="agent verdict matched user feedback",
            id="similar",
        )

        review = DecisionAgent(DecisionProfile(user_id="u1", criteria={})).review(
            request,
            history_records=(*unrelated_records, similar_record),
        )

        self.assertTrue(any("important similar artifact feedback" in issue.reason for issue in review.issues))

    def test_decision_records_round_trip_as_jsonl(self) -> None:
        profile = DecisionProfile(user_id="u1", criteria={})
        request = ArtifactReviewRequest(
            task_type="blog_outline",
            intent="write about Decision Agent",
            artifact="A short outline about Decision Agent.",
        )
        review = DecisionAgent(profile).review(request)
        feedback = UserFeedback(verdict="revise", notes="Needs a sharper opening.")
        learned = DecisionAgent(profile).learn(request, review, feedback)

        with TemporaryDirectory() as directory:
            record_path = f"{directory}/blog_outline.jsonl"
            append_decision_record(record_path, learned.decision_records[-1])

            records = load_decision_records(record_path)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].request.task_type, "blog_outline")
        self.assertEqual(records[0].user_feedback.notes, "Needs a sharper opening.")

    def test_record_ids_are_unique_for_repeated_learns(self) -> None:
        profile = DecisionProfile(user_id="u1", criteria={})
        request = ArtifactReviewRequest(
            task_type="blog_outline",
            intent="write about Decision Agent",
            artifact="A short outline about Decision Agent.",
        )
        review = DecisionAgent(profile).review(request)
        feedback = UserFeedback(verdict="revise", notes="Needs a sharper opening.")

        first = DecisionAgent(profile).learn(request, review, feedback)
        second = DecisionAgent(profile).learn(request, review, feedback)

        self.assertNotEqual(first.decision_records[-1].id, second.decision_records[-1].id)

    def test_load_decision_records_skips_malformed_jsonl_rows(self) -> None:
        profile = DecisionProfile(user_id="u1", criteria={})
        request = ArtifactReviewRequest(
            task_type="blog_outline",
            intent="write about Decision Agent",
            artifact="A short outline about Decision Agent.",
        )
        review = DecisionAgent(profile).review(request)
        feedback = UserFeedback(verdict="revise", notes="Needs a sharper opening.")
        learned = DecisionAgent(profile).learn(request, review, feedback)

        with TemporaryDirectory() as directory:
            record_path = f"{directory}/blog_outline.jsonl"
            with open(record_path, "w", encoding="utf-8") as file:
                file.write("{bad json}\n")
            append_decision_record(record_path, learned.decision_records[-1])

            records = load_decision_records(record_path)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].user_feedback.notes, "Needs a sharper opening.")

    def test_evaluate_reports_alignment_and_profile_updates(self) -> None:
        profile = DecisionProfile(user_id="u1", criteria={})
        mismatch_case = EvaluationCase(
            id="concept-first",
            request=ArtifactReviewRequest(
                task_type="blog_outline",
                intent="write about Decision Agent",
                artifact=(
                    "This outline defines Decision Agent, explains loop engineering, describes "
                    "profiles, and then concludes that better feedback loops improve agent output. "
                    "It is coherent but does not show a concrete user pain before the concept."
                ),
            ),
            user_judgment=UserFeedback(
                verdict="revise",
                notes="Problem framing is weak.",
                core_issues=("problem framing is weak",),
                revision_direction="start with a concrete failure case",
            ),
        )
        aligned_case = EvaluationCase(
            id="too-short",
            request=ArtifactReviewRequest(
                task_type="blog_outline",
                intent="write about Decision Agent",
                artifact="Too short.",
            ),
            user_judgment=UserFeedback(
                verdict="revise",
                notes="Too short to judge.",
                core_issues=("artifact is too short",),
                revision_direction="add enough outline detail",
            ),
        )

        report = DecisionAgent(profile).evaluate((mismatch_case, aligned_case))

        self.assertEqual(report.cases, 2)
        self.assertEqual(report.verdict_accuracy, 0.5)
        self.assertIn("problem framing is weak", report.common_misses)
        concept_first_result = next(result for result in report.case_results if result.id == "concept-first")
        self.assertFalse(concept_first_result.verdict_agreement)
        self.assertTrue(concept_first_result.suggested_profile_updates)

    def test_load_evaluation_cases_reports_malformed_jsonl_rows(self) -> None:
        case = EvaluationCase(
            id="case-1",
            request=ArtifactReviewRequest(
                task_type="blog_outline",
                intent="write about Decision Agent",
                artifact="A short outline.",
            ),
            user_judgment=UserFeedback(verdict="revise", notes="Needs detail."),
        )

        with TemporaryDirectory() as directory:
            case_path = f"{directory}/cases.jsonl"
            with open(case_path, "w", encoding="utf-8") as file:
                file.write("{bad json}\n")
                file.write(json.dumps({"id": "bad-shape", "request": {"task_type": "blog_outline"}}))
                file.write("\n")
                file.write(json.dumps(case.to_dict(), ensure_ascii=False))
                file.write("\n")

            with self.assertRaisesRegex(ValueError, "malformed evaluation case row 1"):
                load_evaluation_cases(case_path)

    def test_user_feedback_accepts_scalar_core_issue(self) -> None:
        feedback = UserFeedback.from_dict(
            {
                "verdict": "revise",
                "core_issues": "problem framing is weak",
            }
        )

        self.assertEqual(feedback.core_issues, ("problem framing is weak",))


if __name__ == "__main__":
    unittest.main()
