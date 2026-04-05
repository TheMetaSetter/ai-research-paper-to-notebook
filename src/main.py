from __future__ import annotations

import json
import hashlib
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import typer

from src.artifact_store import RunArtifactStore, utc_timestamp_string
from src.chunking import build_chunks_from_parsed_paper
from src.config import EnvironmentSettings, RunParameters, load_environment_settings, load_run_parameters
from src.parse_pdf import parse_pdf_into_parsed_paper
from src.schemas import ParsedPaper, RunManifest

app = typer.Typer(help="Generate pedagogy-first notebooks from AI research papers.")


@dataclass
class CommandContext:
    environment_settings: EnvironmentSettings
    run_parameters: RunParameters
    artifact_store: RunArtifactStore
    run_manifest: RunManifest | None


def _slugify_file_stem(file_path: Path) -> str:
    normalized_name = re.sub(r"[^a-z0-9]+", "-", file_path.stem.lower()).strip("-")
    return normalized_name or "paper"


def _compute_file_sha256(file_path: Path) -> str:
    file_hasher = hashlib.sha256()
    with file_path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(8192), b""):
            file_hasher.update(chunk)
    return file_hasher.hexdigest()


def _current_code_version() -> str:
    try:
        completed_process = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "unknown"

    return completed_process.stdout.strip() or "unknown"


def _prepare_run_command_context(pdf_path: Path, params_path: Path) -> CommandContext:
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF path does not exist or is not a file: {pdf_path}")

    environment_settings = load_environment_settings()
    run_parameters = load_run_parameters(params_path)

    paper_slug = _slugify_file_stem(pdf_path)
    run_id = utc_timestamp_string()
    artifact_store = RunArtifactStore.create(runs_root_directory=Path("runs"), paper_slug=paper_slug, run_id=run_id)

    run_manifest = RunManifest(
        run_id=run_id,
        paper_slug=paper_slug,
        source_pdf_path=str(pdf_path.resolve()),
        source_pdf_sha256=_compute_file_sha256(pdf_path),
        params_path=str(params_path.resolve()),
        run_parameters=run_parameters.model_dump(mode="json"),
        planned_notebook_output_path=artifact_store.default_stage_artifact_paths()["notebook"],
        active_model_provenance=None,
        stage_artifact_paths=artifact_store.default_stage_artifact_paths(),
        created_at_utc=run_id,
    )
    artifact_store.write_json_model(stage_name=None, file_name="run_manifest.json", payload=run_manifest)

    return CommandContext(
        environment_settings=environment_settings,
        run_parameters=run_parameters,
        artifact_store=artifact_store,
        run_manifest=run_manifest,
    )


def _prepare_existing_run_context(run_directory: Path) -> CommandContext:
    if not run_directory.exists() or not run_directory.is_dir():
        raise FileNotFoundError(f"Run directory does not exist or is not a directory: {run_directory}")

    environment_settings = load_environment_settings()
    run_parameters = load_run_parameters(Path("params.yaml"))
    artifact_store = RunArtifactStore(run_root_directory=run_directory.resolve())
    artifact_store.ensure_standard_layout()

    manifest_path = artifact_store.run_root_directory / "run_manifest.json"
    run_manifest = None
    if manifest_path.is_file():
        run_manifest = RunManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))

    return CommandContext(
        environment_settings=environment_settings,
        run_parameters=run_parameters,
        artifact_store=artifact_store,
        run_manifest=run_manifest,
    )


@app.command()
def run(
    pdf_path: Path,
    params_path: Path = Path("params.yaml"),
) -> None:
    """Run the full pipeline from PDF to validated notebook."""

    _prepare_run_command_context(pdf_path=pdf_path, params_path=params_path)
    raise NotImplementedError("The full pipeline will be implemented in later phases.")


@app.command()
def parse(pdf_path: Path, params_path: Path = Path("params.yaml")) -> None:
    """Run only PDF parsing and emit stage artifacts."""

    command_context = _prepare_run_command_context(pdf_path=pdf_path, params_path=params_path)
    if command_context.run_manifest is None:
        raise RuntimeError("Run manifest was not created for the parse command.")

    parsed_paper, raw_markdown_text, page_chunks = parse_pdf_into_parsed_paper(
        pdf_path=pdf_path,
        source_pdf_sha256=command_context.run_manifest.source_pdf_sha256 or "",
        run_id=command_context.run_manifest.run_id,
        code_version=_current_code_version(),
    )

    parsed_paper_path = command_context.artifact_store.write_json_model(
        stage_name="parsed_paper",
        file_name="parsed_paper.json",
        payload=parsed_paper,
    )
    raw_markdown_path = command_context.artifact_store.write_text(
        stage_name="parsed_paper",
        file_name="raw_markdown.md",
        text=raw_markdown_text,
    )
    page_chunks_path = command_context.artifact_store.write_text(
        stage_name="parsed_paper",
        file_name="page_chunks.json",
        text=json.dumps(page_chunks, indent=2, ensure_ascii=False),
    )

    updated_artifact_paths = dict(command_context.run_manifest.stage_artifact_paths)
    updated_artifact_paths["parsed_paper"] = str(parsed_paper_path)
    updated_artifact_paths["parsed_paper_raw_markdown"] = str(raw_markdown_path)
    updated_artifact_paths["parsed_paper_page_chunks"] = str(page_chunks_path)

    updated_run_manifest = command_context.run_manifest.model_copy(
        update={
            "stage_artifact_paths": updated_artifact_paths,
        }
    )
    command_context.artifact_store.write_json_model(
        stage_name=None,
        file_name="run_manifest.json",
        payload=updated_run_manifest,
    )

    typer.echo(f"Run directory: {command_context.artifact_store.run_root_directory}")
    typer.echo(f"Parsed paper artifact: {parsed_paper_path}")
    typer.echo(f"Raw markdown artifact: {raw_markdown_path}")
    typer.echo(f"Page chunks artifact: {page_chunks_path}")


@app.command()
def chunk(run_directory: Path) -> None:
    """Build provenance-rich chunks from a parsed paper artifact."""

    command_context = _prepare_existing_run_context(run_directory=run_directory)
    if command_context.run_manifest is None:
        raise RuntimeError("Run manifest is required before chunking can run.")

    parsed_paper_path_string = command_context.run_manifest.stage_artifact_paths.get("parsed_paper")
    if not parsed_paper_path_string:
        raise FileNotFoundError("The chunk stage requires a parsed_paper artifact path in run_manifest.json.")

    parsed_paper_path = Path(parsed_paper_path_string)
    if not parsed_paper_path.is_file():
        raise FileNotFoundError(f"The chunk stage requires parsed_paper.json, but it was not found: {parsed_paper_path}")

    parsed_paper = ParsedPaper.model_validate_json(parsed_paper_path.read_text(encoding="utf-8"))
    paper_chunks = build_chunks_from_parsed_paper(
        parsed_paper=parsed_paper,
        chunk_size_characters=command_context.run_parameters.chunk_size_characters,
        chunk_overlap_characters=command_context.run_parameters.chunk_overlap_characters,
    )

    chunks_path = command_context.artifact_store.write_text(
        stage_name="chunks",
        file_name="chunks.json",
        text=json.dumps([paper_chunk.model_dump(mode="json") for paper_chunk in paper_chunks], indent=2, ensure_ascii=False),
    )

    updated_artifact_paths = dict(command_context.run_manifest.stage_artifact_paths)
    updated_artifact_paths["chunks"] = str(chunks_path)

    updated_run_manifest = command_context.run_manifest.model_copy(update={"stage_artifact_paths": updated_artifact_paths})
    command_context.artifact_store.write_json_model(
        stage_name=None,
        file_name="run_manifest.json",
        payload=updated_run_manifest,
    )

    typer.echo(f"Run directory: {command_context.artifact_store.run_root_directory}")
    typer.echo(f"Chunks artifact: {chunks_path}")


@app.command()
def plan(run_directory: Path) -> None:
    """Run concept extraction, learner capture, and notebook planning."""

    _prepare_existing_run_context(run_directory=run_directory)
    raise NotImplementedError("Notebook planning will be implemented in a later phase.")


@app.command()
def generate(run_directory: Path) -> None:
    """Generate notebook cells and assemble the notebook."""

    _prepare_existing_run_context(run_directory=run_directory)
    raise NotImplementedError("Notebook generation will be implemented in a later phase.")


if __name__ == "__main__":
    app()
