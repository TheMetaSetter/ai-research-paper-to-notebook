from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel


# These locations are fixed early so later phases can regenerate one stage at a time.
STANDARD_STAGE_FILE_NAMES: dict[str, str] = {
    "parsed_paper": "parsed_paper.json",
    "chunks": "chunks.json",
    "concept_graph": "concepts.json",
    "learner_profile": "learner_profile.json",
    "notebook_plan": "notebook_plan.json",
    "validation_report": "validation_report.json",
    "notebook": "final_notebook.ipynb",
}

STANDARD_STAGE_DIRECTORIES: tuple[str, ...] = (
    "parsed_paper",
    "chunks",
    "concept_graph",
    "learner_profile",
    "notebook_plan",
    "cell_batches",
    "validation_report",
    "notebook",
)


@dataclass
class RunArtifactStore:
    run_root_directory: Path

    @classmethod
    def create(
        cls,
        runs_root_directory: str | Path,
        paper_slug: str,
        run_id: str,
    ) -> "RunArtifactStore":
        run_root_directory = Path(runs_root_directory) / paper_slug / run_id
        run_root_directory.mkdir(parents=True, exist_ok=True)
        artifact_store = cls(run_root_directory=run_root_directory)
        artifact_store.ensure_standard_layout()
        return artifact_store

    def ensure_standard_layout(self) -> None:
        for stage_name in STANDARD_STAGE_DIRECTORIES:
            self.stage_directory(stage_name)

    def stage_directory(self, stage_name: str) -> Path:
        stage_path = self.run_root_directory / stage_name
        stage_path.mkdir(parents=True, exist_ok=True)
        return stage_path

    def list_stage_files(self, stage_name: str, glob_pattern: str) -> list[Path]:
        """Return deterministically ordered stage files for a narrow stage helper."""

        return sorted(self.stage_directory(stage_name).glob(glob_pattern))

    def default_stage_artifact_paths(self) -> dict[str, str]:
        return {
            stage_name: str(self.stage_directory(stage_name) / file_name)
            for stage_name, file_name in STANDARD_STAGE_FILE_NAMES.items()
        } | {
            "cell_batches": str(self.stage_directory("cell_batches")),
            "run_manifest": str(self.run_root_directory / "run_manifest.json"),
        }

    def write_json_model(
        self,
        stage_name: str | None,
        file_name: str,
        payload: BaseModel,
    ) -> Path:
        return self.write_json_dict(
            stage_name=stage_name,
            file_name=file_name,
            payload=payload.model_dump(mode="json"),
        )

    def write_json_dict(
        self,
        stage_name: str | None,
        file_name: str,
        payload: dict[str, Any],
    ) -> Path:
        destination_path = self._destination_path(stage_name=stage_name, file_name=file_name)
        with destination_path.open("w", encoding="utf-8") as output_file:
            json.dump(payload, output_file, indent=2, ensure_ascii=False)
        return destination_path

    def write_text(self, stage_name: str | None, file_name: str, text: str) -> Path:
        destination_path = self._destination_path(stage_name=stage_name, file_name=file_name)
        destination_path.write_text(text, encoding="utf-8")
        return destination_path

    def _destination_path(self, stage_name: str | None, file_name: str) -> Path:
        if stage_name is None:
            return self.run_root_directory / file_name
        return self.stage_directory(stage_name) / file_name


def utc_timestamp_string() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
