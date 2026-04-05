from __future__ import annotations

import pytest

from src.chunking import (
    build_chunks_from_parsed_paper,
    extract_equation_markers,
    extract_figure_references,
    extract_notation_tokens,
    split_text_with_overlap,
)
from src.retrieve import PaperChunkRetriever, RetrievalHit, tokenize_for_bm25
from src.schemas import PaperChunk, ParsedPaper, PaperSection


def _build_sample_parsed_paper() -> ParsedPaper:
    return ParsedPaper(
        paper_id="paper",
        paper_title="Attention Title",
        source_pdf_path="/tmp/paper.pdf",
        source_pdf_sha256="abc123",
        parser_name="pymupdf4llm",
        parser_version="0.0.20",
        parsed_at_utc="20260405T120000Z",
        sections=[
            PaperSection(
                section_id="section_000",
                title="Introduction",
                page_start=1,
                page_end=1,
                markdown_text="Attention uses Q = XW_Q. Figure 1 shows the model.",
            ),
            PaperSection(
                section_id="section_001",
                title="Method",
                page_start=2,
                page_end=3,
                markdown_text="We compute K and V. Table 2 reports results.",
            ),
            PaperSection(
                section_id="section_002",
                title="Empty",
                page_start=4,
                page_end=4,
                markdown_text="   ",
            ),
        ],
        abstract_text="",
        authors=None,
        stage_provenance=None,
    )


def test_chunk_overlap_preserves_boundary_context() -> None:
    original_text = "abcdefghijklmnopqrstuvwxyz"
    chunks = split_text_with_overlap(
        full_text=original_text,
        chunk_size_characters=10,
        chunk_overlap_characters=3,
    )
    assert chunks == ["abcdefghij", "hijklmnopq", "opqrstuvwx", "vwxyz"]


def test_chunking_rejects_invalid_overlap_configuration() -> None:
    with pytest.raises(ValueError, match="smaller than chunk_size_characters"):
        split_text_with_overlap(
            full_text="abcdef",
            chunk_size_characters=5,
            chunk_overlap_characters=5,
        )


def test_build_chunks_from_parsed_paper_preserves_provenance_and_skips_empty_sections() -> None:
    paper_chunks = build_chunks_from_parsed_paper(
        parsed_paper=_build_sample_parsed_paper(),
        chunk_size_characters=100,
        chunk_overlap_characters=10,
    )

    assert [paper_chunk.chunk_id for paper_chunk in paper_chunks] == ["chunk_00000", "chunk_00001"]
    assert paper_chunks[0].section_id == "section_000"
    assert paper_chunks[0].section_title == "Introduction"
    assert paper_chunks[0].page_start == 1
    assert paper_chunks[0].page_end == 1
    assert paper_chunks[1].section_id == "section_001"


def test_chunk_metadata_extractors_are_deterministic() -> None:
    section_fragment = r"Q = XW_Q and \frac{1}{\sqrt{d_k}}. Figure 1 and Table 2 appear."

    assert extract_equation_markers(section_fragment) == ["=", r"\frac"]
    assert extract_notation_tokens(section_fragment) == ["Q", "XW_Q", "d_k"]
    assert extract_figure_references(section_fragment) == ["Figure 1", "Table 2"]


def test_tokenize_for_bm25_is_stable() -> None:
    assert tokenize_for_bm25("Attention, Q/K/V!") == ["attention", "q", "k", "v"]


def test_retriever_requires_rank_bm25(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.retrieve.BM25Okapi", None)

    with pytest.raises(RuntimeError, match="requires rank_bm25"):
        PaperChunkRetriever(paper_chunks=[])


def test_retriever_prefers_section_and_priority_boosts(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeBM25Okapi:
        def __init__(self, tokenized_chunk_texts: list[list[str]]) -> None:
            self.tokenized_chunk_texts = tokenized_chunk_texts

        def get_scores(self, tokenized_query: list[str]) -> list[float]:
            return [1.0, 1.0, 1.0]

    monkeypatch.setattr("src.retrieve.BM25Okapi", FakeBM25Okapi)

    paper_chunks = [
        PaperChunk(
            chunk_id="chunk_00002",
            section_id="section_002",
            section_title="Method",
            page_start=3,
            page_end=3,
            chunk_text="Generic text.",
            equation_markers=[],
            notation_tokens=[],
            figure_references=[],
        ),
        PaperChunk(
            chunk_id="chunk_00000",
            section_id="section_000",
            section_title="Introduction",
            page_start=1,
            page_end=1,
            chunk_text="Attention uses Q and K.",
            equation_markers=["="],
            notation_tokens=["attention", "q", "k"],
            figure_references=[],
        ),
        PaperChunk(
            chunk_id="chunk_00001",
            section_id="section_001",
            section_title="Abstract",
            page_start=2,
            page_end=2,
            chunk_text="Attention overview.",
            equation_markers=[],
            notation_tokens=["attention"],
            figure_references=[],
        ),
    ]

    retriever = PaperChunkRetriever(paper_chunks=paper_chunks)
    retrieval_hits = retriever.search_with_scores(
        query_text="attention q equation",
        top_k=3,
        preferred_section_id="section_000",
    )

    assert isinstance(retrieval_hits[0], RetrievalHit)
    assert [retrieval_hit.paper_chunk.chunk_id for retrieval_hit in retrieval_hits] == [
        "chunk_00000",
        "chunk_00001",
        "chunk_00002",
    ]
    assert retrieval_hits[0].final_score > retrieval_hits[1].final_score > retrieval_hits[2].final_score


def test_retriever_search_returns_chunks_only(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeBM25Okapi:
        def __init__(self, tokenized_chunk_texts: list[list[str]]) -> None:
            self.tokenized_chunk_texts = tokenized_chunk_texts

        def get_scores(self, tokenized_query: list[str]) -> list[float]:
            return [0.1, 0.1]

    monkeypatch.setattr("src.retrieve.BM25Okapi", FakeBM25Okapi)

    paper_chunks = [
        PaperChunk(
            chunk_id="chunk_00001",
            section_id="section_001",
            section_title="Method",
            page_start=2,
            page_end=2,
            chunk_text="Method text.",
            equation_markers=[],
            notation_tokens=[],
            figure_references=[],
        ),
        PaperChunk(
            chunk_id="chunk_00000",
            section_id="section_000",
            section_title="Introduction",
            page_start=1,
            page_end=1,
            chunk_text="Introduction text.",
            equation_markers=[],
            notation_tokens=[],
            figure_references=[],
        ),
    ]

    retriever = PaperChunkRetriever(paper_chunks=paper_chunks)
    search_results = retriever.search(query_text="text", top_k=2)

    assert [paper_chunk.chunk_id for paper_chunk in search_results] == ["chunk_00000", "chunk_00001"]
