from __future__ import annotations

from pathlib import Path

import pytest

from src.parse_pdf import parse_pdf_into_parsed_paper, split_markdown_pages_into_sections


class FakePyMuPDF4LLM:
    def __init__(self, payload):
        self.payload = payload
        self.calls: list[dict[str, object]] = []

    def to_markdown(self, path: str, *, page_chunks: bool, page_separators: bool):
        self.calls.append(
            {
                "path": path,
                "page_chunks": page_chunks,
                "page_separators": page_separators,
            }
        )
        return self.payload


def _sample_page_chunks() -> list[dict[str, object]]:
    return [
        {
            "metadata": {"page_number": 1, "file_path": "/tmp/paper.pdf", "page_count": 2},
            "text": "Alice Author\nInstitute\n\n# Paper Title\nIntro paragraph.\n## Abstract\nAbstract line one.",
        },
        {
            "metadata": {"page_number": 2, "file_path": "/tmp/paper.pdf", "page_count": 2},
            "text": "Abstract line two.\n## Method\nMethod section text.",
        },
    ]


def test_split_markdown_pages_into_sections_preserves_front_matter_and_page_ranges() -> None:
    normalized_page_chunks = [
        {"page_number": 1, "text": "Alice Author\nInstitute\n\n# Paper Title\nIntro paragraph.\n## Abstract\nAbstract line one."},
        {"page_number": 2, "text": "Abstract line two.\n## Method\nMethod section text."},
    ]

    sections = split_markdown_pages_into_sections(normalized_page_chunks)

    assert [section.section_id for section in sections] == ["section_000", "section_001", "section_002", "section_003"]
    assert sections[0].title == "Front Matter"
    assert sections[0].markdown_text == "Alice Author\nInstitute"
    assert sections[0].page_start == 1
    assert sections[0].page_end == 1
    assert sections[2].title == "Abstract"
    assert sections[2].page_start == 1
    assert sections[2].page_end == 2
    assert sections[2].markdown_text == "Abstract line one.\nAbstract line two."


def test_parse_pdf_into_parsed_paper_builds_expected_contract(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_parser = FakePyMuPDF4LLM(_sample_page_chunks())
    monkeypatch.setattr("src.parse_pdf.pymupdf4llm", fake_parser)
    monkeypatch.setattr("src.parse_pdf._parser_version_string", lambda: "0.0.20")

    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    parsed_paper, raw_markdown_text, normalized_page_chunks = parse_pdf_into_parsed_paper(
        pdf_path=pdf_path,
        source_pdf_sha256="abc123",
        run_id="20260405T120000Z",
        code_version="test-code-version",
    )

    assert fake_parser.calls == [
        {
            "path": str(pdf_path),
            "page_chunks": True,
            "page_separators": False,
        }
    ]
    assert parsed_paper.paper_id == "paper"
    assert parsed_paper.paper_title == "Paper Title"
    assert parsed_paper.abstract_text == "Abstract line one.\nAbstract line two."
    assert parsed_paper.parser_name == "pymupdf4llm"
    assert parsed_paper.parser_version == "0.0.20"
    assert parsed_paper.stage_provenance is not None
    assert parsed_paper.stage_provenance.stage_name == "parsed_paper"
    assert parsed_paper.stage_provenance.code_version == "test-code-version"
    assert raw_markdown_text.startswith("Alice Author")
    assert normalized_page_chunks[0]["page_number"] == 1


def test_parse_pdf_into_parsed_paper_uses_pdf_stem_when_no_h1_exists(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_parser = FakePyMuPDF4LLM(
        [
            {
                "metadata": {"page_number": 1, "file_path": "/tmp/no-title.pdf", "page_count": 1},
                "text": "No markdown title here.\nJust body text.",
            }
        ]
    )
    monkeypatch.setattr("src.parse_pdf.pymupdf4llm", fake_parser)

    pdf_path = tmp_path / "no-title.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    parsed_paper, _, _ = parse_pdf_into_parsed_paper(
        pdf_path=pdf_path,
        source_pdf_sha256="abc123",
        run_id="20260405T120000Z",
    )

    assert parsed_paper.paper_title == "no-title"
    assert parsed_paper.abstract_text == ""


def test_parse_pdf_into_parsed_paper_rejects_unsupported_parser_shape(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_parser = FakePyMuPDF4LLM(payload="not-a-page-chunk-list")
    monkeypatch.setattr("src.parse_pdf.pymupdf4llm", fake_parser)

    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    with pytest.raises(RuntimeError, match="unsupported page chunk response"):
        parse_pdf_into_parsed_paper(
            pdf_path=pdf_path,
            source_pdf_sha256="abc123",
            run_id="20260405T120000Z",
        )


def test_parse_pdf_into_parsed_paper_requires_pymupdf4llm(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("src.parse_pdf.pymupdf4llm", None)

    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    with pytest.raises(RuntimeError, match="requires pymupdf4llm"):
        parse_pdf_into_parsed_paper(
            pdf_path=pdf_path,
            source_pdf_sha256="abc123",
            run_id="20260405T120000Z",
        )
