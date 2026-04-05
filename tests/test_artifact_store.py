from __future__ import annotations

import json
import re
from pathlib import Path

from src.artifact_store import RunArtifactStore, utc_timestamp_string
from src.schemas import LearnerProfile, RunManifest


def test_create_sets_up_locked_stage_directories(tmp_path: Path) -> None:
    artifact_store = RunArtifactStore.create(
        runs_root_directory=tmp_path / "runs",
        paper_slug="attention-is-all-you-need",
        run_id="20260405T120000Z",
    )

    expected_directories = [
        "parsed_paper",
        "chunks",
        "concept_graph",
        "learner_profile",
        "notebook_plan",
        "cell_batches",
        "validation_report",
        "notebook",
    ]

    for directory_name in expected_directories:
        assert (artifact_store.run_root_directory / directory_name).is_dir()


def test_write_json_model_and_dict_write_expected_files(tmp_path: Path) -> None:
    artifact_store = RunArtifactStore.create(
        runs_root_directory=tmp_path / "runs",
        paper_slug="paper",
        run_id="20260405T120000Z",
    )

    learner_profile_path = artifact_store.write_json_model(
        stage_name="learner_profile",
        file_name="learner_profile.json",
        payload=LearnerProfile(
            mathematics_background="Math",
            machine_learning_background="ML",
            deep_learning_background="DL",
            python_background="Python",
            tensor_familiarity="Intermediate",
        ),
    )
    manifest_path = artifact_store.write_json_model(
        stage_name=None,
        file_name="run_manifest.json",
        payload=RunManifest(
            run_id="20260405T120000Z",
            paper_slug="paper",
            source_pdf_path="/tmp/paper.pdf",
            source_pdf_sha256="abc123",
            params_path="/tmp/params.yaml",
            run_parameters={"inference_backend": "llama_cpp"},
            stage_artifact_paths=artifact_store.default_stage_artifact_paths(),
            created_at_utc="20260405T120000Z",
        ),
    )
    concept_graph_path = artifact_store.write_json_dict(
        stage_name="concept_graph",
        file_name="concepts.json",
        payload={"concepts": [{"concept_id": "concept_1"}]},
    )

    assert learner_profile_path == artifact_store.run_root_directory / "learner_profile" / "learner_profile.json"
    assert concept_graph_path == artifact_store.run_root_directory / "concept_graph" / "concepts.json"
    assert manifest_path == artifact_store.run_root_directory / "run_manifest.json"

    learner_profile_payload = json.loads(learner_profile_path.read_text(encoding="utf-8"))
    concept_payload = json.loads(concept_graph_path.read_text(encoding="utf-8"))

    assert learner_profile_payload["mathematics_background"] == "Math"
    assert concept_payload["concepts"][0]["concept_id"] == "concept_1"


def test_default_stage_artifact_paths_match_locked_layout(tmp_path: Path) -> None:
    artifact_store = RunArtifactStore.create(
        runs_root_directory=tmp_path / "runs",
        paper_slug="paper",
        run_id="20260405T120000Z",
    )

    stage_paths = artifact_store.default_stage_artifact_paths()

    assert stage_paths["parsed_paper"].endswith("parsed_paper/parsed_paper.json")
    assert stage_paths["chunks"].endswith("chunks/chunks.json")
    assert stage_paths["concept_graph"].endswith("concept_graph/concepts.json")
    assert stage_paths["notebook"].endswith("notebook/final_notebook.ipynb")
    assert stage_paths["run_manifest"].endswith("run_manifest.json")


def test_utc_timestamp_string_is_stable_utc_shape() -> None:
    timestamp_string = utc_timestamp_string()

    assert re.fullmatch(r"\d{8}T\d{6}Z", timestamp_string)

