from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from decision_agent.models import (
    Alternative,
    ArtifactReview,
    ArtifactReviewRequest,
    DecisionProfile,
    DecisionRecord,
    UserFeedback,
)


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


def load_decision_records(path: str | Path) -> tuple[DecisionRecord, ...]:
    record_path = Path(path)
    if not record_path.exists():
        return ()

    records: list[DecisionRecord] = []
    with record_path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(DecisionRecord.from_dict(json.loads(line)))
    return tuple(records)


def append_decision_record(path: str | Path, record: DecisionRecord) -> None:
    record_path = Path(path)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    with record_path.open("a", encoding="utf-8") as file:
        json.dump(record.to_dict(), file, ensure_ascii=False)
        file.write("\n")


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _save_json(data: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
        file.write("\n")
