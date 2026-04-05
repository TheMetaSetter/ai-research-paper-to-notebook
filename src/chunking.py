from __future__ import annotations

import re

from src.schemas import PaperChunk, ParsedPaper


EQUATION_PATTERN = re.compile(r"\\\(|\\\)|\\\[|\\\]|=|\\frac|\\sum|\\mathbb|\\mathbf")
NOTATION_CANDIDATE_PATTERN = re.compile(r"\b[A-Za-z][A-Za-z0-9_]*\b")
FIGURE_REFERENCE_PATTERN = re.compile(r"\b(?:Figure|Fig\.|Table)\s+\d+\b")


def split_text_with_overlap(
    full_text: str,
    chunk_size_characters: int,
    chunk_overlap_characters: int,
) -> list[str]:
    if chunk_size_characters <= 0:
        raise ValueError("chunk_size_characters must be positive.")
    if chunk_overlap_characters < 0:
        raise ValueError("chunk_overlap_characters must be non-negative.")
    if chunk_overlap_characters >= chunk_size_characters:
        raise ValueError("chunk_overlap_characters must be smaller than chunk_size_characters.")

    text_fragments: list[str] = []
    start_index = 0

    while start_index < len(full_text):
        end_index = min(len(full_text), start_index + chunk_size_characters)
        text_fragments.append(full_text[start_index:end_index])
        if end_index == len(full_text):
            break
        start_index = end_index - chunk_overlap_characters

    return text_fragments


def extract_equation_markers(section_fragment: str) -> list[str]:
    return EQUATION_PATTERN.findall(section_fragment)


def extract_notation_tokens(section_fragment: str) -> list[str]:
    notation_tokens: set[str] = set()
    for notation_candidate in NOTATION_CANDIDATE_PATTERN.findall(section_fragment):
        if "_" in notation_candidate:
            notation_tokens.add(notation_candidate)
            continue

        if len(notation_candidate) == 1 and notation_candidate.isalpha():
            notation_tokens.add(notation_candidate)
            continue

        if notation_candidate.isupper() and len(notation_candidate) <= 4:
            notation_tokens.add(notation_candidate)

    return sorted(notation_tokens)


def extract_figure_references(section_fragment: str) -> list[str]:
    return FIGURE_REFERENCE_PATTERN.findall(section_fragment)


def build_chunks_from_parsed_paper(
    parsed_paper: ParsedPaper,
    chunk_size_characters: int,
    chunk_overlap_characters: int,
) -> list[PaperChunk]:
    generated_chunks: list[PaperChunk] = []

    for paper_section in parsed_paper.sections:
        trimmed_section_text = paper_section.markdown_text.strip()
        if not trimmed_section_text:
            continue

        section_fragments = split_text_with_overlap(
            full_text=trimmed_section_text,
            chunk_size_characters=chunk_size_characters,
            chunk_overlap_characters=chunk_overlap_characters,
        )

        for section_fragment in section_fragments:
            chunk_index = len(generated_chunks)
            generated_chunks.append(
                PaperChunk(
                    chunk_id=f"chunk_{chunk_index:05d}",
                    section_id=paper_section.section_id,
                    section_title=paper_section.title,
                    page_start=paper_section.page_start,
                    page_end=paper_section.page_end,
                    chunk_text=section_fragment,
                    equation_markers=extract_equation_markers(section_fragment),
                    notation_tokens=extract_notation_tokens(section_fragment),
                    figure_references=extract_figure_references(section_fragment),
                )
            )

    return generated_chunks
