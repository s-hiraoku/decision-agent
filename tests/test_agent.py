import unittest
from tempfile import TemporaryDirectory

from decision_agent.agent import DecisionAgent
from decision_agent.models import (
    Alternative,
    ArtifactReviewRequest,
    DecisionExample,
    DecisionProfile,
    UserFeedback,
)
from decision_agent.storage import append_decision_record, load_decision_records


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


if __name__ == "__main__":
    unittest.main()
