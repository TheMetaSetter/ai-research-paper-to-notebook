from __future__ import annotations

import re
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from src.artifact_store import utc_timestamp_string
from src.schemas import ParsedPaper, PaperSection, StageProvenance

try:
    import pymupdf4llm
except Exception:  # pragma: no cover - runtime dependency is optional in tests
    pymupdf4llm = None


SECTION_HEADER_PATTERN = re.compile(r"^(#{1,6})\s+(.*)$")


def _slugify_file_stem(file_path: Path) -> str:
    normalized_name = re.sub(r"[^a-z0-9]+", "-", file_path.stem.lower()).strip("-")
    return normalized_name or "paper"


def _require_pymupdf4llm() -> Any:
    if pymupdf4llm is None:
        raise RuntimeError(
            "The parse stage requires pymupdf4llm, but it is not installed. "
            "Install the 'pymupdf4llm' package to run PDF ingestion."
        )
    return pymupdf4llm


def _parser_version_string() -> str:
    try:
        return version("pymupdf4llm")
    except PackageNotFoundError:
        return "unknown"


def _normalize_page_chunks(raw_page_chunks: Any) -> list[dict[str, object]]:
    if not isinstance(raw_page_chunks, list):
        raise RuntimeError("PyMuPDF4LLM returned an unsupported page chunk response: expected a list of page dictionaries.")

    normalized_page_chunks: list[dict[str, object]] = []
    for page_chunk in raw_page_chunks:
        if not isinstance(page_chunk, dict):
            raise RuntimeError("PyMuPDF4LLM returned an unsupported page chunk response: each page chunk must be a dictionary.")

        metadata = page_chunk.get("metadata")
        text = page_chunk.get("text")
        if not isinstance(metadata, dict) or not isinstance(text, str):
            raise RuntimeError(
                "PyMuPDF4LLM returned an unsupported page chunk response: each page chunk must contain 'metadata' and 'text'."
            )

        page_number = metadata.get("page_number")
        if not isinstance(page_number, int):
            raise RuntimeError(
                "PyMuPDF4LLM returned an unsupported page chunk response: page chunk metadata must contain integer 'page_number'."
            )

        normalized_page_chunks.append(
            {
                "page_number": page_number,
                "text": text,
                "metadata": metadata,
            }
        )

    return normalized_page_chunks


def _finalize_section(
    collected_sections: list[PaperSection],
    current_section_title: str | None,
    current_section_lines: list[str],
    current_page_start: int | None,
    current_page_end: int | None,
) -> None:
    if current_section_title is None:
        return

    section_markdown_text = "\n".join(current_section_lines).strip()
    if not section_markdown_text:
        return

    collected_sections.append(
        PaperSection(
            section_id=f"section_{len(collected_sections):03d}",
            title=current_section_title,
            page_start=current_page_start,
            page_end=current_page_end,
            markdown_text=section_markdown_text,
        )
    )


def _first_h1_title(page_chunks: list[dict[str, object]]) -> str | None:
    for page_chunk in page_chunks:
        page_text = page_chunk["text"]
        if not isinstance(page_text, str):
            continue
        for markdown_line in page_text.splitlines():
            header_match = SECTION_HEADER_PATTERN.match(markdown_line)
            if header_match is not None and header_match.group(1) == "#":
                return header_match.group(2).strip()
    return None


def split_markdown_pages_into_sections(page_chunks: list[dict[str, object]]) -> list[PaperSection]:
    collected_sections: list[PaperSection] = []
    current_section_title: str | None = "Front Matter"
    current_section_lines: list[str] = []
    current_page_start: int | None = None
    current_page_end: int | None = None

    for page_chunk in page_chunks:
        page_number = page_chunk.get("page_number")
        page_text = page_chunk.get("text")
        if not isinstance(page_number, int) or not isinstance(page_text, str):
            raise RuntimeError("Normalized page chunks must contain integer 'page_number' and string 'text'.")

        for markdown_line in page_text.splitlines():
            header_match = SECTION_HEADER_PATTERN.match(markdown_line)
            if header_match is not None:
                _finalize_section(
                    collected_sections=collected_sections,
                    current_section_title=current_section_title,
                    current_section_lines=current_section_lines,
                    current_page_start=current_page_start,
                    current_page_end=current_page_end,
                )
                current_section_title = header_match.group(2).strip()
                current_section_lines = []
                current_page_start = page_number
                current_page_end = page_number
                continue

            if current_page_start is None:
                current_page_start = page_number
            current_page_end = page_number
            current_section_lines.append(markdown_line)

    _finalize_section(
        collected_sections=collected_sections,
        current_section_title=current_section_title,
        current_section_lines=current_section_lines,
        current_page_start=current_page_start,
        current_page_end=current_page_end,
    )

    return collected_sections


def parse_pdf_into_parsed_paper(
    pdf_path: str | Path,
    source_pdf_sha256: str,
    run_id: str,
    code_version: str | None = None,
) -> tuple[ParsedPaper, str, list[dict[str, object]]]:
    parser_module = _require_pymupdf4llm()
    pdf_path = Path(pdf_path)

    raw_page_chunks = parser_module.to_markdown(
        str(pdf_path),
        page_chunks=True,
        page_separators=False,
    )
    normalized_page_chunks = _normalize_page_chunks(raw_page_chunks)
    paper_sections = split_markdown_pages_into_sections(normalized_page_chunks)
    raw_markdown_text = "\n\n".join(str(page_chunk["text"]).strip() for page_chunk in normalized_page_chunks).strip()

    paper_title = _first_h1_title(normalized_page_chunks) or pdf_path.stem
    abstract_text = ""
    for paper_section in paper_sections:
        if paper_section.title.casefold() == "abstract":
            abstract_text = paper_section.markdown_text
            break

    parsed_paper = ParsedPaper(
        paper_id=_slugify_file_stem(pdf_path),
        paper_title=paper_title,
        source_pdf_path=str(pdf_path.resolve()),
        source_pdf_sha256=source_pdf_sha256,
        parser_name="pymupdf4llm",
        parser_version=_parser_version_string(),
        parsed_at_utc=utc_timestamp_string(),
        sections=paper_sections,
        abstract_text=abstract_text,
        authors=None,
        stage_provenance=StageProvenance(
            stage_name="parsed_paper",
            created_at_utc=utc_timestamp_string(),
            input_artifact_paths=[str(pdf_path.resolve())],
            output_artifact_path="parsed_paper.json",
            code_version=code_version or "unknown",
            model_provenance=None,
        ),
    )
    return parsed_paper, raw_markdown_text, normalized_page_chunks
