from __future__ import annotations

import json
import hashlib
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import nbformat
import typer

from src.artifact_store import RunArtifactStore, utc_timestamp_string
from src.chunking import build_chunks_from_parsed_paper
from src.cell_generator import generate_notebook_batch_for_section, save_notebook_batch, select_retrieved_chunks_for_lesson_section
from src.concept_graph import build_concept_graph
from src.config import EnvironmentSettings, RunParameters, load_environment_settings, load_run_parameters
from src.learner_profile import capture_learner_profile_interactively, save_learner_profile
from src.notebook_builder import build_notebook_metadata, build_notebook_object, load_notebook_batches, write_notebook
from src.models.gemma_4_e2b import Gemma4E2BModel
from src.parse_pdf import parse_pdf_into_parsed_paper
from src.planner import build_notebook_plan, save_notebook_plan
from src.repair import attempt_targeted_repair
from src.schemas import ConceptItem, LearnerProfile, NotebookBatch, NotebookPlan, PaperChunk, ParsedPaper, RunManifest
from src.validators import load_notebook_object, validate_assembled_notebook

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
def concept(run_directory: Path) -> None:
    """Extract and merge concept graph artifacts from chunked paper content."""

    command_context = _prepare_existing_run_context(run_directory=run_directory)
    if command_context.run_manifest is None:
        raise RuntimeError("Run manifest is required before concept extraction can run.")

    chunks_path_string = command_context.run_manifest.stage_artifact_paths.get("chunks")
    if not chunks_path_string:
        raise FileNotFoundError("The concept stage requires a chunks artifact path in run_manifest.json.")

    chunks_path = Path(chunks_path_string)
    if not chunks_path.is_file():
        raise FileNotFoundError(f"The concept stage requires chunks.json, but it was not found: {chunks_path}")

    paper_chunks = [PaperChunk.model_validate(chunk_payload) for chunk_payload in json.loads(chunks_path.read_text(encoding="utf-8"))]
    concept_model = Gemma4E2BModel.from_settings(
        environment_settings=command_context.environment_settings,
        run_parameters=command_context.run_parameters,
    )
    concept_items, concept_edges = build_concept_graph(
        model=concept_model,
        paper_chunks=paper_chunks,
        retrieval_top_k=command_context.run_parameters.retrieval_top_k,
    )

    concepts_path = command_context.artifact_store.write_text(
        stage_name="concept_graph",
        file_name="concepts.json",
        text=json.dumps([concept_item.model_dump(mode="json") for concept_item in concept_items], indent=2, ensure_ascii=False),
    )
    concept_edges_path = command_context.artifact_store.write_text(
        stage_name="concept_graph",
        file_name="concept_edges.json",
        text=json.dumps(concept_edges, indent=2, ensure_ascii=False),
    )

    updated_artifact_paths = dict(command_context.run_manifest.stage_artifact_paths)
    updated_artifact_paths["concept_graph"] = str(concepts_path)
    updated_artifact_paths["concept_graph_edges"] = str(concept_edges_path)
    updated_run_manifest = command_context.run_manifest.model_copy(update={"stage_artifact_paths": updated_artifact_paths})
    command_context.artifact_store.write_json_model(
        stage_name=None,
        file_name="run_manifest.json",
        payload=updated_run_manifest,
    )

    typer.echo(f"Run directory: {command_context.artifact_store.run_root_directory}")
    typer.echo(f"Concept graph artifact: {concepts_path}")
    typer.echo(f"Concept edges artifact: {concept_edges_path}")


@app.command()
def plan(run_directory: Path) -> None:
    """Capture or reuse a learner profile and build the notebook plan."""

    command_context = _prepare_existing_run_context(run_directory=run_directory)
    if command_context.run_manifest is None:
        raise RuntimeError("Run manifest is required before learner profile capture can run.")

    concept_graph_path_string = command_context.run_manifest.stage_artifact_paths.get("concept_graph")
    if not concept_graph_path_string:
        raise FileNotFoundError("The learner profile stage requires a concept_graph artifact path in run_manifest.json.")

    concept_graph_path = Path(concept_graph_path_string)
    if not concept_graph_path.is_file():
        raise FileNotFoundError(
            f"The learner profile stage requires concepts.json, but it was not found: {concept_graph_path}"
        )

    learner_profile_path_string = command_context.run_manifest.stage_artifact_paths.get("learner_profile")
    learner_profile_path = Path(learner_profile_path_string) if learner_profile_path_string else None
    if learner_profile_path is not None and learner_profile_path.is_file():
        learner_profile = LearnerProfile.model_validate_json(learner_profile_path.read_text(encoding="utf-8"))
    else:
        learner_profile = capture_learner_profile_interactively()
        learner_profile_path = save_learner_profile(
            learner_profile=learner_profile,
            output_path=command_context.artifact_store.stage_directory("learner_profile") / "learner_profile.json",
        )

    concept_items = [
        ConceptItem.model_validate(concept_item_payload)
        for concept_item_payload in json.loads(concept_graph_path.read_text(encoding="utf-8"))
    ]
    parsed_paper_path_string = command_context.run_manifest.stage_artifact_paths.get("parsed_paper")
    paper_id = command_context.run_manifest.paper_slug
    if parsed_paper_path_string:
        parsed_paper_path = Path(parsed_paper_path_string)
        if parsed_paper_path.is_file():
            paper_id = ParsedPaper.model_validate_json(parsed_paper_path.read_text(encoding="utf-8")).paper_id

    planning_model = Gemma4E2BModel.from_settings(
        environment_settings=command_context.environment_settings,
        run_parameters=command_context.run_parameters,
    )
    notebook_plan_path = command_context.artifact_store.stage_directory("notebook_plan") / "notebook_plan.json"
    notebook_plan = build_notebook_plan(
        model=planning_model,
        learner_profile=learner_profile,
        concept_items=concept_items,
        paper_id=paper_id,
        input_artifact_paths=[str(concept_graph_path), str(learner_profile_path)],
        output_artifact_path=str(notebook_plan_path),
        code_version=_current_code_version(),
        maximum_generation_sections=command_context.run_parameters.maximum_generation_sections,
    )
    saved_notebook_plan_path = save_notebook_plan(notebook_plan=notebook_plan, output_path=notebook_plan_path)

    updated_artifact_paths = dict(command_context.run_manifest.stage_artifact_paths)
    updated_artifact_paths["learner_profile"] = str(learner_profile_path)
    updated_artifact_paths["notebook_plan"] = str(saved_notebook_plan_path)
    updated_run_manifest = command_context.run_manifest.model_copy(
        update={
            "active_model_provenance": planning_model.model_provenance(),
            "stage_artifact_paths": updated_artifact_paths,
        }
    )
    command_context.artifact_store.write_json_model(
        stage_name=None,
        file_name="run_manifest.json",
        payload=updated_run_manifest,
    )

    typer.echo(f"Run directory: {command_context.artifact_store.run_root_directory}")
    typer.echo(f"Learner profile artifact: {learner_profile_path}")
    typer.echo(f"Notebook plan artifact: {saved_notebook_plan_path}")


@app.command()
def generate(run_directory: Path) -> None:
    """Generate validated section-wise notebook cell batches."""

    command_context = _prepare_existing_run_context(run_directory=run_directory)
    if command_context.run_manifest is None:
        raise RuntimeError("Run manifest is required before notebook generation can run.")

    chunks_path_string = command_context.run_manifest.stage_artifact_paths.get("chunks")
    if not chunks_path_string:
        raise FileNotFoundError("The generate stage requires a chunks artifact path in run_manifest.json.")

    notebook_plan_path_string = command_context.run_manifest.stage_artifact_paths.get("notebook_plan")
    if not notebook_plan_path_string:
        raise FileNotFoundError("The generate stage requires a notebook_plan artifact path in run_manifest.json.")

    learner_profile_path_string = command_context.run_manifest.stage_artifact_paths.get("learner_profile")
    if not learner_profile_path_string:
        raise FileNotFoundError("The generate stage requires a learner_profile artifact path in run_manifest.json.")

    chunks_path = Path(chunks_path_string)
    notebook_plan_path = Path(notebook_plan_path_string)
    learner_profile_path = Path(learner_profile_path_string)
    if not chunks_path.is_file():
        raise FileNotFoundError(f"The generate stage requires chunks.json, but it was not found: {chunks_path}")
    if not notebook_plan_path.is_file():
        raise FileNotFoundError(
            f"The generate stage requires notebook_plan.json, but it was not found: {notebook_plan_path}"
        )
    if not learner_profile_path.is_file():
        raise FileNotFoundError(
            f"The generate stage requires learner_profile.json, but it was not found: {learner_profile_path}"
        )

    paper_chunks = [PaperChunk.model_validate(chunk_payload) for chunk_payload in json.loads(chunks_path.read_text(encoding="utf-8"))]
    notebook_plan = NotebookPlan.model_validate_json(notebook_plan_path.read_text(encoding="utf-8"))
    learner_profile = LearnerProfile.model_validate_json(learner_profile_path.read_text(encoding="utf-8"))
    generation_model = Gemma4E2BModel.from_settings(
        environment_settings=command_context.environment_settings,
        run_parameters=command_context.run_parameters,
    )

    lesson_sections = notebook_plan.lesson_sections
    if command_context.run_parameters.maximum_generation_sections is not None:
        lesson_sections = lesson_sections[: command_context.run_parameters.maximum_generation_sections]

    saved_notebook_batch_paths: list[Path] = []
    for lesson_section in lesson_sections:
        retrieved_chunks = select_retrieved_chunks_for_lesson_section(
            lesson_section=lesson_section,
            paper_chunks=paper_chunks,
            retrieval_top_k=command_context.run_parameters.retrieval_top_k,
        )
        notebook_batch_path = command_context.artifact_store.stage_directory("cell_batches") / f"{lesson_section.section_id}.json"
        notebook_batch = generate_notebook_batch_for_section(
            model=generation_model,
            learner_profile=learner_profile,
            lesson_section=lesson_section,
            retrieved_chunks=retrieved_chunks,
            input_artifact_paths=[str(notebook_plan_path), str(learner_profile_path), str(chunks_path)],
            output_artifact_path=str(notebook_batch_path),
            code_version=_current_code_version(),
        )
        saved_notebook_batch_paths.append(
            save_notebook_batch(notebook_batch=notebook_batch, output_path=notebook_batch_path)
        )

    updated_artifact_paths = dict(command_context.run_manifest.stage_artifact_paths)
    updated_artifact_paths["cell_batches"] = str(command_context.artifact_store.stage_directory("cell_batches"))
    for saved_notebook_batch_path in saved_notebook_batch_paths:
        updated_artifact_paths[f"cell_batch_{saved_notebook_batch_path.stem}"] = str(saved_notebook_batch_path)

    updated_run_manifest = command_context.run_manifest.model_copy(
        update={
            "active_model_provenance": generation_model.model_provenance(),
            "stage_artifact_paths": updated_artifact_paths,
        }
    )
    command_context.artifact_store.write_json_model(
        stage_name=None,
        file_name="run_manifest.json",
        payload=updated_run_manifest,
    )

    typer.echo(f"Run directory: {command_context.artifact_store.run_root_directory}")
    for saved_notebook_batch_path in saved_notebook_batch_paths:
        typer.echo(f"Cell batch artifact: {saved_notebook_batch_path}")


@app.command()
def assemble(run_directory: Path) -> None:
    """Assemble saved section batches into a final notebook artifact."""

    command_context = _prepare_existing_run_context(run_directory=run_directory)
    if command_context.run_manifest is None:
        raise RuntimeError("Run manifest is required before notebook assembly can run.")

    notebook_plan_path_string = command_context.run_manifest.stage_artifact_paths.get("notebook_plan")
    if not notebook_plan_path_string:
        raise FileNotFoundError("The assemble stage requires a notebook_plan artifact path in run_manifest.json.")

    learner_profile_path_string = command_context.run_manifest.stage_artifact_paths.get("learner_profile")
    if not learner_profile_path_string:
        raise FileNotFoundError("The assemble stage requires a learner_profile artifact path in run_manifest.json.")

    notebook_plan_path = Path(notebook_plan_path_string)
    learner_profile_path = Path(learner_profile_path_string)
    if not notebook_plan_path.is_file():
        raise FileNotFoundError(
            f"The assemble stage requires notebook_plan.json, but it was not found: {notebook_plan_path}"
        )
    if not learner_profile_path.is_file():
        raise FileNotFoundError(
            f"The assemble stage requires learner_profile.json, but it was not found: {learner_profile_path}"
        )

    cell_batch_paths = command_context.artifact_store.list_stage_files("cell_batches", "*.json")
    if not cell_batch_paths:
        raise FileNotFoundError("The assemble stage requires at least one cell batch JSON artifact in cell_batches/.")

    notebook_plan = NotebookPlan.model_validate_json(notebook_plan_path.read_text(encoding="utf-8"))
    learner_profile = LearnerProfile.model_validate_json(learner_profile_path.read_text(encoding="utf-8"))
    loaded_notebook_batches = load_notebook_batches(cell_batch_paths)

    planned_section_ids = [lesson_section.section_id for lesson_section in notebook_plan.lesson_sections]
    notebook_batches_by_section_id: dict[str, NotebookBatch] = {}
    for notebook_batch in loaded_notebook_batches:
        if notebook_batch.section_id in notebook_batches_by_section_id:
            raise RuntimeError(f"Duplicate notebook batch detected for section_id={notebook_batch.section_id}.")
        notebook_batches_by_section_id[notebook_batch.section_id] = notebook_batch

    unknown_batch_section_ids = sorted(set(notebook_batches_by_section_id) - set(planned_section_ids))
    if unknown_batch_section_ids:
        raise RuntimeError(
            "Notebook assembly found batch sections missing from the notebook plan: "
            + ", ".join(unknown_batch_section_ids)
        )

    missing_planned_section_ids = [section_id for section_id in planned_section_ids if section_id not in notebook_batches_by_section_id]
    if missing_planned_section_ids:
        raise RuntimeError(
            "Notebook assembly is missing saved batches for planned sections: " + ", ".join(missing_planned_section_ids)
        )

    ordered_notebook_batches = [notebook_batches_by_section_id[section_id] for section_id in planned_section_ids]

    parsed_paper = None
    parsed_paper_path_string = command_context.run_manifest.stage_artifact_paths.get("parsed_paper")
    if parsed_paper_path_string:
        parsed_paper_path = Path(parsed_paper_path_string)
        if parsed_paper_path.is_file():
            parsed_paper = ParsedPaper.model_validate_json(parsed_paper_path.read_text(encoding="utf-8"))

    notebook_title = parsed_paper.paper_title if parsed_paper is not None else notebook_plan.paper_id
    notebook_metadata = build_notebook_metadata(
        notebook_plan=notebook_plan,
        run_id=command_context.run_manifest.run_id,
        code_version=_current_code_version(),
        model_provenance=command_context.run_manifest.active_model_provenance,
        generated_section_batches=ordered_notebook_batches,
        parsed_paper=parsed_paper,
    )
    notebook_object = build_notebook_object(
        notebook_title=notebook_title,
        learner_profile=learner_profile,
        generated_section_batches=ordered_notebook_batches,
        notebook_metadata=notebook_metadata,
    )

    notebook_output_path = write_notebook(
        notebook_object=notebook_object,
        output_path=command_context.artifact_store.stage_directory("notebook") / "final_notebook.ipynb",
    )

    updated_artifact_paths = dict(command_context.run_manifest.stage_artifact_paths)
    updated_artifact_paths["notebook"] = str(notebook_output_path)
    updated_run_manifest = command_context.run_manifest.model_copy(update={"stage_artifact_paths": updated_artifact_paths})
    command_context.artifact_store.write_json_model(
        stage_name=None,
        file_name="run_manifest.json",
        payload=updated_run_manifest,
    )

    typer.echo(f"Run directory: {command_context.artifact_store.run_root_directory}")
    typer.echo(f"Notebook artifact: {notebook_output_path}")


@app.command()
def validate(run_directory: Path) -> None:
    """Validate the assembled notebook and optionally run targeted repairs."""

    command_context = _prepare_existing_run_context(run_directory=run_directory)
    if command_context.run_manifest is None:
        raise RuntimeError("Run manifest is required before notebook validation can run.")

    notebook_path_string = command_context.run_manifest.stage_artifact_paths.get("notebook")
    if not notebook_path_string:
        raise FileNotFoundError("The validate stage requires a notebook artifact path in run_manifest.json.")

    notebook_plan_path_string = command_context.run_manifest.stage_artifact_paths.get("notebook_plan")
    if not notebook_plan_path_string:
        raise FileNotFoundError("The validate stage requires a notebook_plan artifact path in run_manifest.json.")

    learner_profile_path_string = command_context.run_manifest.stage_artifact_paths.get("learner_profile")
    if not learner_profile_path_string:
        raise FileNotFoundError("The validate stage requires a learner_profile artifact path in run_manifest.json.")

    notebook_path = Path(notebook_path_string)
    notebook_plan_path = Path(notebook_plan_path_string)
    learner_profile_path = Path(learner_profile_path_string)
    if not notebook_path.is_file():
        raise FileNotFoundError(f"The validate stage requires final_notebook.ipynb, but it was not found: {notebook_path}")
    if not notebook_plan_path.is_file():
        raise FileNotFoundError(
            f"The validate stage requires notebook_plan.json, but it was not found: {notebook_plan_path}"
        )
    if not learner_profile_path.is_file():
        raise FileNotFoundError(
            f"The validate stage requires learner_profile.json, but it was not found: {learner_profile_path}"
        )

    cell_batch_paths = command_context.artifact_store.list_stage_files("cell_batches", "*.json")
    generated_section_batches = load_notebook_batches(cell_batch_paths)
    notebook_object = load_notebook_object(notebook_path)
    notebook_plan = NotebookPlan.model_validate_json(notebook_plan_path.read_text(encoding="utf-8"))
    learner_profile = LearnerProfile.model_validate_json(learner_profile_path.read_text(encoding="utf-8"))

    parsed_paper = None
    parsed_paper_path_string = command_context.run_manifest.stage_artifact_paths.get("parsed_paper")
    if parsed_paper_path_string:
        parsed_paper_path = Path(parsed_paper_path_string)
        if parsed_paper_path.is_file():
            parsed_paper = ParsedPaper.model_validate_json(parsed_paper_path.read_text(encoding="utf-8"))

    validation_report, validation_issues = validate_assembled_notebook(
        notebook_object=notebook_object,
        notebook_plan=notebook_plan,
        learner_profile=learner_profile,
        generated_section_batches=generated_section_batches,
        paper_id=notebook_plan.paper_id,
        run_id=command_context.run_manifest.run_id,
        notebook_path=str(notebook_path),
        input_artifact_paths=[
            str(notebook_path),
            str(notebook_plan_path),
            str(learner_profile_path),
            *[str(cell_batch_path) for cell_batch_path in cell_batch_paths],
        ],
        code_version=_current_code_version(),
        enable_smoke_test=command_context.run_parameters.enable_notebook_execution_smoke_test,
        execution_timeout_seconds=command_context.run_parameters.notebook_execution_timeout_seconds,
        working_directory=str(command_context.artifact_store.run_root_directory),
    )

    if not validation_report.is_valid and command_context.run_parameters.enable_repair_pass:
        repair_outcome = attempt_targeted_repair(
            validation_issues=validation_issues,
            artifact_store=command_context.artifact_store,
            run_manifest=command_context.run_manifest,
            notebook_plan=notebook_plan,
            learner_profile=learner_profile,
            generated_section_batches=generated_section_batches,
            parsed_paper=parsed_paper,
            environment_settings=command_context.environment_settings,
            run_parameters=command_context.run_parameters,
            code_version=_current_code_version(),
        )
        if repair_outcome.repaired:
            repaired_notebook_path = Path(repair_outcome.repaired_notebook_path) if repair_outcome.repaired_notebook_path else notebook_path
            reloaded_notebook_object = nbformat.read(repaired_notebook_path, as_version=4)
            repaired_batch_paths = command_context.artifact_store.list_stage_files("cell_batches", "*.json")
            repaired_section_batches = load_notebook_batches(repaired_batch_paths)
            validation_report, validation_issues = validate_assembled_notebook(
                notebook_object=reloaded_notebook_object,
                notebook_plan=notebook_plan,
                learner_profile=learner_profile,
                generated_section_batches=repaired_section_batches,
                paper_id=notebook_plan.paper_id,
                run_id=command_context.run_manifest.run_id,
                notebook_path=str(repaired_notebook_path),
                input_artifact_paths=[
                    str(repaired_notebook_path),
                    str(notebook_plan_path),
                    str(learner_profile_path),
                    *[str(repaired_batch_path) for repaired_batch_path in repaired_batch_paths],
                ],
                code_version=_current_code_version(),
                enable_smoke_test=command_context.run_parameters.enable_notebook_execution_smoke_test,
                execution_timeout_seconds=command_context.run_parameters.notebook_execution_timeout_seconds,
                working_directory=str(command_context.artifact_store.run_root_directory),
            )

    validation_report_path = command_context.artifact_store.write_json_model(
        stage_name="validation_report",
        file_name="validation_report.json",
        payload=validation_report,
    )

    updated_artifact_paths = dict(command_context.run_manifest.stage_artifact_paths)
    updated_artifact_paths["validation_report"] = str(validation_report_path)
    updated_run_manifest = command_context.run_manifest.model_copy(update={"stage_artifact_paths": updated_artifact_paths})
    command_context.artifact_store.write_json_model(
        stage_name=None,
        file_name="run_manifest.json",
        payload=updated_run_manifest,
    )

    typer.echo(f"Run directory: {command_context.artifact_store.run_root_directory}")
    typer.echo(f"Validation report artifact: {validation_report_path}")


if __name__ == "__main__":
    app()
