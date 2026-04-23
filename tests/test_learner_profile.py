from __future__ import annotations

from pathlib import Path

from src.learner_profile import _prompt_choice, _prompt_required_text, capture_learner_profile_interactively, save_learner_profile
from src.schemas import LearnerProfile


def test_capture_learner_profile_interactively_returns_valid_profile(monkeypatch) -> None:
    prompt_responses = iter(
        [
            "Linear algebra",
            "Intermediate ML",
            "Beginner DL",
            "Intermediate Python",
            "Comfortable with matrix multiplication",
            "DEEP",
            "Moderate",
        ]
    )
    confirm_responses = iter([True, False])

    monkeypatch.setattr("src.learner_profile.typer.prompt", lambda prompt_text, default=None: next(prompt_responses))
    monkeypatch.setattr("src.learner_profile.typer.confirm", lambda prompt_text, default=True: next(confirm_responses))

    learner_profile = capture_learner_profile_interactively()

    assert learner_profile.mathematics_background == "Linear algebra"
    assert learner_profile.wants_tensor_shapes is True
    assert learner_profile.wants_derivations is False
    assert learner_profile.preferred_depth == "deep"
    assert learner_profile.preferred_pacing == "moderate"


def test_required_text_prompt_reprompts_until_non_blank(monkeypatch) -> None:
    prompt_responses = iter(["  ", " Linear algebra "])
    echoed_messages: list[str] = []

    monkeypatch.setattr("src.learner_profile.typer.prompt", lambda prompt_text: next(prompt_responses))
    monkeypatch.setattr("src.learner_profile.typer.echo", echoed_messages.append)

    response_text = _prompt_required_text("Mathematics background")

    assert response_text == "Linear algebra"
    assert echoed_messages == ["Mathematics background cannot be blank."]


def test_choice_prompt_reprompts_until_valid_choice(monkeypatch) -> None:
    prompt_responses = iter(["mediumish", " DEEP "])
    echoed_messages: list[str] = []

    monkeypatch.setattr("src.learner_profile.typer.prompt", lambda prompt_text, default=None: next(prompt_responses))
    monkeypatch.setattr("src.learner_profile.typer.echo", echoed_messages.append)

    response_text = _prompt_choice("Depth: light / medium / deep", ("light", "medium", "deep"), default="deep")

    assert response_text == "deep"
    assert echoed_messages == ["Depth: light / medium / deep must be one of: light, medium, deep."]


def test_save_learner_profile_round_trips_json(tmp_path: Path) -> None:
    learner_profile = LearnerProfile(
        mathematics_background="Linear algebra",
        machine_learning_background="Intermediate ML",
        deep_learning_background="Beginner DL",
        python_background="Intermediate Python",
        tensor_familiarity="Comfortable with matrix multiplication",
        wants_tensor_shapes=True,
        wants_derivations=False,
        preferred_depth="deep",
        preferred_pacing="moderate",
    )

    learner_profile_path = save_learner_profile(
        learner_profile=learner_profile,
        output_path=tmp_path / "learner_profile" / "learner_profile.json",
    )

    reloaded_learner_profile = LearnerProfile.model_validate_json(learner_profile_path.read_text(encoding="utf-8"))
    assert reloaded_learner_profile == learner_profile
