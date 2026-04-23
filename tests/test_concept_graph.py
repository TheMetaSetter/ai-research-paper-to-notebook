from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.concept_graph import (
    ConceptCandidate,
    build_concept_edges,
    build_concept_graph,
    build_supporting_context_chunks,
    merge_concept_candidates,
)
from src.schemas import ConceptItem, PaperChunk


def _sample_paper_chunks() -> list[PaperChunk]:
    return [
        PaperChunk(
            chunk_id="chunk_00000",
            section_id="section_000",
            section_title="Introduction",
            page_start=1,
            page_end=1,
            chunk_text="Attention introduces Q and K.",
            equation_markers=["="],
            notation_tokens=["attention", "q", "k"],
            figure_references=[],
        ),
        PaperChunk(
            chunk_id="chunk_00001",
            section_id="section_000",
            section_title="Introduction",
            page_start=1,
            page_end=1,
            chunk_text="Softmax normalizes the attention scores.",
            equation_markers=[],
            notation_tokens=["softmax"],
            figure_references=[],
        ),
        PaperChunk(
            chunk_id="chunk_00002",
            section_id="section_001",
            section_title="Method",
            page_start=2,
            page_end=2,
            chunk_text="The loss function is minimized.",
            equation_markers=[],
            notation_tokens=["loss"],
            figure_references=[],
        ),
    ]


def test_build_supporting_context_chunks_uses_same_section_retrieval(monkeypatch: pytest.MonkeyPatch) -> None:
    expected_support_chunk = _sample_paper_chunks()[1]

    class FakeRetriever:
        def __init__(self, paper_chunks: list[PaperChunk]) -> None:
            self.paper_chunks = paper_chunks

        def search(self, query_text: str, top_k: int = 6, preferred_section_id: str | None = None) -> list[PaperChunk]:
            assert preferred_section_id == "section_000"
            assert top_k == 2
            return [expected_support_chunk]

    monkeypatch.setattr("src.concept_graph.PaperChunkRetriever", FakeRetriever)

    support_chunks = build_supporting_context_chunks(
        paper_chunks=_sample_paper_chunks(),
        primary_chunk=_sample_paper_chunks()[0],
        retrieval_top_k=3,
    )

    assert support_chunks == [expected_support_chunk]


def test_merge_concept_candidates_is_deterministic_and_creates_synthetic_prerequisites() -> None:
    concept_candidates = [
        ConceptCandidate(
            label="Scaled Dot-Product Attention",
            concept_type="equation",
            source_chunk_ids=["chunk_00000"],
            prerequisites=["Softmax", "Linear Algebra"],
            equation_text="softmax(QK^T)V",
            notation_symbols=["Q", "K"],
            tensor_shape_notes=["Q: B x T x d_k"],
            confidence=0.6,
        ),
        ConceptCandidate(
            label="scaled   dot-product attention",
            concept_type="equation",
            source_chunk_ids=["chunk_00001"],
            prerequisites=["softmax"],
            equation_text="softmax(QK^T / sqrt(d_k))V",
            notation_symbols=["V"],
            tensor_shape_notes=["V: B x T x d_v"],
            confidence=0.9,
        ),
        ConceptCandidate(
            label="Softmax",
            concept_type="definition",
            source_chunk_ids=["chunk_00001"],
            prerequisites=[],
            equation_text=None,
            notation_symbols=[],
            tensor_shape_notes=[],
            confidence=0.8,
        ),
    ]

    merged_concepts = merge_concept_candidates(concept_candidates)
    merged_by_label = {concept_item.label: concept_item for concept_item in merged_concepts}

    attention_concept = merged_by_label["Scaled Dot-Product Attention"]
    softmax_concept = merged_by_label["Softmax"]
    linear_algebra_concept = merged_by_label["Linear Algebra"]

    assert attention_concept.concept_id == "concept_001"
    assert attention_concept.source_chunk_ids == ["chunk_00000", "chunk_00001"]
    assert attention_concept.equation_text == "softmax(QK^T / sqrt(d_k))V"
    assert attention_concept.notation_symbols == ["K", "Q", "V"]
    assert attention_concept.tensor_shape_notes == ["Q: B x T x d_k", "V: B x T x d_v"]
    assert attention_concept.confidence == 0.9
    assert attention_concept.prerequisites == [softmax_concept.concept_id, linear_algebra_concept.concept_id]
    assert linear_algebra_concept.concept_type == "prerequisite"


def test_build_concept_edges_orders_prerequisite_edges_stably() -> None:
    concept_items = [
        ConceptItem(
            concept_id="concept_001",
            label="Attention",
            concept_type="equation",
            source_chunk_ids=["chunk_00000"],
            prerequisites=["concept_000", "concept_002"],
            equation_text=None,
            notation_symbols=[],
            tensor_shape_notes=[],
            confidence=None,
        ),
        ConceptItem(
            concept_id="concept_000",
            label="Softmax",
            concept_type="definition",
            source_chunk_ids=["chunk_00001"],
            prerequisites=[],
            equation_text=None,
            notation_symbols=[],
            tensor_shape_notes=[],
            confidence=None,
        ),
    ]

    concept_edges = build_concept_edges(concept_items)

    assert concept_edges == [
        {"source_concept_id": "concept_000", "target_concept_id": "concept_001"},
        {"source_concept_id": "concept_002", "target_concept_id": "concept_001"},
    ]


def test_build_concept_graph_processes_chunks_in_order_and_merges_results() -> None:
    @dataclass
    class FakeModel:
        outputs_by_chunk_id: dict[str, list[ConceptCandidate]]

    fake_model = FakeModel(
        outputs_by_chunk_id={
            "chunk_00000": [
                ConceptCandidate(
                    label="Attention",
                    concept_type="equation",
                    source_chunk_ids=["chunk_00000"],
                    prerequisites=["Softmax"],
                    equation_text="softmax(QK^T)V",
                    notation_symbols=["Q", "K"],
                    tensor_shape_notes=["Q: B x T x d_k"],
                    confidence=0.8,
                )
            ],
            "chunk_00001": [
                ConceptCandidate(
                    label="Softmax",
                    concept_type="definition",
                    source_chunk_ids=["chunk_00001"],
                    prerequisites=[],
                    equation_text=None,
                    notation_symbols=[],
                    tensor_shape_notes=[],
                    confidence=0.7,
                )
            ],
            "chunk_00002": [],
        }
    )

    monkeypatch_calls: list[tuple[str, list[str]]] = []

    def fake_build_supporting_context_chunks(
        paper_chunks: list[PaperChunk],
        primary_chunk: PaperChunk,
        retrieval_top_k: int,
    ) -> list[PaperChunk]:
        return [paper_chunks[1]] if primary_chunk.chunk_id == "chunk_00000" else []

    def fake_extract_concept_candidates_from_chunk(
        model,
        primary_chunk: PaperChunk,
        supporting_chunks: list[PaperChunk],
    ) -> list[ConceptCandidate]:
        monkeypatch_calls.append((primary_chunk.chunk_id, [supporting_chunk.chunk_id for supporting_chunk in supporting_chunks]))
        return model.outputs_by_chunk_id[primary_chunk.chunk_id]

    import src.concept_graph as concept_graph_module

    original_support_builder = concept_graph_module.build_supporting_context_chunks
    original_extractor = concept_graph_module.extract_concept_candidates_from_chunk
    concept_graph_module.build_supporting_context_chunks = fake_build_supporting_context_chunks
    concept_graph_module.extract_concept_candidates_from_chunk = fake_extract_concept_candidates_from_chunk
    try:
        concept_items, concept_edges = build_concept_graph(
            model=fake_model,
            paper_chunks=_sample_paper_chunks(),
            retrieval_top_k=3,
        )
    finally:
        concept_graph_module.build_supporting_context_chunks = original_support_builder
        concept_graph_module.extract_concept_candidates_from_chunk = original_extractor

    assert monkeypatch_calls == [
        ("chunk_00000", ["chunk_00001"]),
        ("chunk_00001", []),
        ("chunk_00002", []),
    ]
    assert [concept_item.label for concept_item in concept_items] == ["Softmax", "Attention"]
    assert concept_edges == [{"source_concept_id": "concept_000", "target_concept_id": "concept_001"}]
