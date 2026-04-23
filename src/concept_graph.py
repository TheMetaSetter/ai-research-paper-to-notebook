from __future__ import annotations

import re
from collections import defaultdict

from pydantic import BaseModel, ConfigDict, Field

from src.models.gemma_4_e2b import Gemma4E2BModel
from src.retrieve import PaperChunkRetriever
from src.schemas import ConceptItem, PaperChunk


CONCEPT_EXTRACTION_SYSTEM_INSTRUCTION = """
You extract teaching concepts from a research-paper chunk.
Return JSON only.
Use officially recognized academic terminology.
Identify prerequisites explicitly by label.
State tensor-shape-relevant information whenever present.
Keep source-faithful wording and do not invent hidden notebook sections.
""".strip()


def normalize_concept_label(concept_label: str) -> str:
    return re.sub(r"\s+", " ", concept_label.strip().casefold())


class ConceptCandidate(ConceptItem):
    concept_id: str = "candidate"
    prerequisites: list[str] = Field(default_factory=list)
    source_chunk_ids: list[str]


class ConceptExtractionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    concepts: list[ConceptCandidate]


def build_supporting_context_chunks(
    paper_chunks: list[PaperChunk],
    primary_chunk: PaperChunk,
    retrieval_top_k: int,
) -> list[PaperChunk]:
    if retrieval_top_k <= 1:
        return []

    same_section_chunks = [
        paper_chunk
        for paper_chunk in paper_chunks
        if paper_chunk.section_id == primary_chunk.section_id and paper_chunk.chunk_id != primary_chunk.chunk_id
    ]
    if not same_section_chunks:
        return []

    retriever = PaperChunkRetriever(same_section_chunks)
    query_text = f"{primary_chunk.section_title}\n{primary_chunk.chunk_text}"
    return retriever.search(
        query_text=query_text,
        top_k=retrieval_top_k - 1,
        preferred_section_id=primary_chunk.section_id,
    )


def extract_concept_candidates_from_chunk(
    model: Gemma4E2BModel,
    primary_chunk: PaperChunk,
    supporting_chunks: list[PaperChunk],
) -> list[ConceptCandidate]:
    supporting_context_text = "\n\n".join(
        [
            (
                f"Supporting Chunk ID: {supporting_chunk.chunk_id}\n"
                f"Supporting Section Title: {supporting_chunk.section_title}\n"
                f"Supporting Text:\n{supporting_chunk.chunk_text}"
            )
            for supporting_chunk in supporting_chunks
        ]
    ).strip()

    user_prompt_sections = [
        "Extract concept candidates from this research-paper chunk.",
        f"Primary Chunk ID: {primary_chunk.chunk_id}",
        f"Primary Section Title: {primary_chunk.section_title}",
        "Primary Text:",
        primary_chunk.chunk_text,
    ]
    if supporting_context_text:
        user_prompt_sections.extend(
            [
                "",
                "Same-section supporting context:",
                supporting_context_text,
            ]
        )

    response = model.generate_json(
        system_instruction=CONCEPT_EXTRACTION_SYSTEM_INSTRUCTION,
        user_prompt="\n".join(user_prompt_sections).strip(),
        response_model=ConceptExtractionResponse,
    )
    return response.concepts


def merge_concept_candidates(concept_candidates: list[ConceptCandidate]) -> list[ConceptItem]:
    grouped_candidates: dict[tuple[str, str], list[ConceptCandidate]] = defaultdict(list)
    for concept_candidate in concept_candidates:
        merge_key = (concept_candidate.concept_type, normalize_concept_label(concept_candidate.label))
        grouped_candidates[merge_key].append(concept_candidate)

    sorted_merge_keys = sorted(grouped_candidates)
    concept_ids_by_key = {merge_key: f"concept_{index:03d}" for index, merge_key in enumerate(sorted_merge_keys)}

    merged_concepts: list[ConceptItem] = []
    unresolved_prerequisite_labels: set[str] = set()

    for merge_key in sorted_merge_keys:
        grouped_items = grouped_candidates[merge_key]
        source_chunk_ids = sorted({chunk_id for item in grouped_items for chunk_id in item.source_chunk_ids})
        notation_symbols = sorted({symbol for item in grouped_items for symbol in item.notation_symbols})
        tensor_shape_notes = sorted({note for item in grouped_items for note in item.tensor_shape_notes})
        equation_text_candidates = [item.equation_text for item in grouped_items if item.equation_text]
        prerequisite_labels = sorted({label for item in grouped_items for label in item.prerequisites if label.strip()})
        unresolved_prerequisite_labels.update(prerequisite_labels)
        confidence_candidates = [item.confidence for item in grouped_items if item.confidence is not None]
        canonical_label = min(grouped_items, key=lambda item: (len(item.label), item.label.casefold())).label.strip()

        merged_concepts.append(
            ConceptItem(
                concept_id=concept_ids_by_key[merge_key],
                label=canonical_label,
                concept_type=grouped_items[0].concept_type,
                source_chunk_ids=source_chunk_ids,
                prerequisites=prerequisite_labels,
                equation_text=max(equation_text_candidates, key=len) if equation_text_candidates else None,
                notation_symbols=notation_symbols,
                tensor_shape_notes=tensor_shape_notes,
                confidence=max(confidence_candidates) if confidence_candidates else None,
            )
        )

    existing_labels_by_normalized_form = {
        normalize_concept_label(concept_item.label): concept_item.concept_id for concept_item in merged_concepts
    }
    for unresolved_label in sorted(unresolved_prerequisite_labels):
        normalized_label = normalize_concept_label(unresolved_label)
        if normalized_label not in existing_labels_by_normalized_form:
            synthetic_concept_id = f"concept_{len(merged_concepts):03d}"
            synthetic_concept = ConceptItem(
                concept_id=synthetic_concept_id,
                label=unresolved_label.strip(),
                concept_type="prerequisite",
                source_chunk_ids=[],
                prerequisites=[],
                equation_text=None,
                notation_symbols=[],
                tensor_shape_notes=[],
                confidence=None,
            )
            merged_concepts.append(synthetic_concept)
            existing_labels_by_normalized_form[normalized_label] = synthetic_concept_id

    resolved_concepts: list[ConceptItem] = []
    for concept_item in merged_concepts:
        resolved_prerequisites = sorted(
            {
                existing_labels_by_normalized_form[normalize_concept_label(prerequisite_label)]
                for prerequisite_label in concept_item.prerequisites
                if normalize_concept_label(prerequisite_label) in existing_labels_by_normalized_form
            }
        )
        resolved_concepts.append(concept_item.model_copy(update={"prerequisites": resolved_prerequisites}))

    return resolved_concepts


def build_concept_edges(concept_items: list[ConceptItem]) -> list[dict[str, str]]:
    concept_edges = [
        {
            "source_concept_id": prerequisite_concept_id,
            "target_concept_id": concept_item.concept_id,
        }
        for concept_item in concept_items
        for prerequisite_concept_id in concept_item.prerequisites
    ]
    return sorted(concept_edges, key=lambda concept_edge: (concept_edge["source_concept_id"], concept_edge["target_concept_id"]))


def build_concept_graph(
    model: Gemma4E2BModel,
    paper_chunks: list[PaperChunk],
    retrieval_top_k: int,
) -> tuple[list[ConceptItem], list[dict[str, str]]]:
    concept_candidates: list[ConceptCandidate] = []

    for primary_chunk in sorted(paper_chunks, key=lambda paper_chunk: paper_chunk.chunk_id):
        supporting_chunks = build_supporting_context_chunks(
            paper_chunks=paper_chunks,
            primary_chunk=primary_chunk,
            retrieval_top_k=retrieval_top_k,
        )
        concept_candidates.extend(
            extract_concept_candidates_from_chunk(
                model=model,
                primary_chunk=primary_chunk,
                supporting_chunks=supporting_chunks,
            )
        )

    concept_items = merge_concept_candidates(concept_candidates)
    concept_edges = build_concept_edges(concept_items)
    return concept_items, concept_edges
