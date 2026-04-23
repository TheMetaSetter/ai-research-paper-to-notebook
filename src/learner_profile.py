from __future__ import annotations

from pathlib import Path

import typer

from src.schemas import LearnerProfile


DEPTH_CHOICES = ("light", "medium", "deep")
PACING_CHOICES = ("fast", "moderate", "slow")


def _prompt_required_text(prompt_text: str) -> str:
    while True:
        response_text = str(typer.prompt(prompt_text)).strip()
        if response_text:
            return response_text

        typer.echo(f"{prompt_text} cannot be blank.")


def _prompt_choice(prompt_text: str, valid_choices: tuple[str, ...], default: str) -> str:
    valid_choice_set = set(valid_choices)
    while True:
        response_text = str(typer.prompt(prompt_text, default=default)).strip().casefold()
        if response_text in valid_choice_set:
            return response_text

        typer.echo(f"{prompt_text} must be one of: {', '.join(valid_choices)}.")


def capture_learner_profile_interactively() -> LearnerProfile:
    return LearnerProfile(
        mathematics_background=_prompt_required_text("Mathematics background"),
        machine_learning_background=_prompt_required_text("Machine learning background"),
        deep_learning_background=_prompt_required_text("Deep learning background"),
        python_background=_prompt_required_text("Python background"),
        tensor_familiarity=_prompt_required_text("Tensor familiarity"),
        wants_tensor_shapes=typer.confirm("Do you want explicit tensor shapes?", default=True),
        wants_derivations=typer.confirm("Do you want mathematical derivations?", default=True),
        preferred_depth=_prompt_choice("Depth: light / medium / deep", DEPTH_CHOICES, default="deep"),
        preferred_pacing=_prompt_choice("Pacing: fast / moderate / slow", PACING_CHOICES, default="moderate"),
    )


def save_learner_profile(learner_profile: LearnerProfile, output_path: str | Path) -> Path:
    learner_profile_path = Path(output_path)
    learner_profile_path.parent.mkdir(parents=True, exist_ok=True)
    learner_profile_path.write_text(learner_profile.model_dump_json(indent=2), encoding="utf-8")
    return learner_profile_path
