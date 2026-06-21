from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from decision_agent.models import Alternative, ArtifactReview, ArtifactReviewRequest, DecisionProfile, UserFeedback


def load_profile(path: str | Path) -> DecisionProfile:
    return DecisionProfile.from_dict(_load_json(path))


def save_profile(profile: DecisionProfile, path: str | Path) -> None:
    _save_json(profile.to_dict(), path)


def load_request(path: str | Path) -> tuple[str, list[Alternative]]:
    data = _load_json(path)
    context = str(data.get("context", ""))
    alternatives = [Alternative.from_dict(item) for item in data.get("alternatives", [])]
    return context, alternatives


def load_review_request(path: str | Path) -> ArtifactReviewRequest:
    return ArtifactReviewRequest.from_dict(_load_json(path))


def load_review(path: str | Path) -> ArtifactReview:
    return ArtifactReview.from_dict(_load_json(path))


def load_feedback(path: str | Path) -> UserFeedback:
    return UserFeedback.from_dict(_load_json(path))


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _save_json(data: dict[str, Any], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
        file.write("\n")
