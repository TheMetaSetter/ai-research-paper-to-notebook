from __future__ import annotations

import re
from dataclasses import dataclass

from src.schemas import PaperChunk

try:
    from rank_bm25 import BM25Okapi
except Exception:  # pragma: no cover - runtime dependency can be unavailable in tests
    BM25Okapi = None


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")
MATH_QUERY_TOKENS = {"q", "k", "v", "loss", "equation", "formula", "attention"}
PRIORITY_SECTION_TITLES = {"title", "abstract", "introduction"}


def _require_rank_bm25() -> type[BM25Okapi]:
    if BM25Okapi is None:
        raise RuntimeError(
            "The retrieval stage requires rank_bm25, but it is not installed. "
            "Install the 'rank-bm25' package to use lexical retrieval."
        )
    return BM25Okapi


def tokenize_for_bm25(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


@dataclass(frozen=True)
class RetrievalHit:
    paper_chunk: PaperChunk
    bm25_score: float
    heuristic_score: float
    final_score: float


class PaperChunkRetriever:
    def __init__(self, paper_chunks: list[PaperChunk]) -> None:
        bm25_class = _require_rank_bm25()
        self.paper_chunks = paper_chunks
        self.tokenized_chunk_texts = [tokenize_for_bm25(paper_chunk.chunk_text) for paper_chunk in paper_chunks]
        self.bm25_index = bm25_class(self.tokenized_chunk_texts)

    def search_with_scores(
        self,
        query_text: str,
        top_k: int = 6,
        preferred_section_id: str | None = None,
    ) -> list[RetrievalHit]:
        tokenized_query = tokenize_for_bm25(query_text)
        bm25_scores = self.bm25_index.get_scores(tokenized_query)

        retrieval_hits: list[RetrievalHit] = []
        for paper_chunk, bm25_score in zip(self.paper_chunks, bm25_scores, strict=False):
            heuristic_score = self._heuristic_score(
                paper_chunk=paper_chunk,
                query_tokens=tokenized_query,
                preferred_section_id=preferred_section_id,
            )
            retrieval_hits.append(
                RetrievalHit(
                    paper_chunk=paper_chunk,
                    bm25_score=float(bm25_score),
                    heuristic_score=heuristic_score,
                    final_score=float(bm25_score) + heuristic_score,
                )
            )

        sorted_hits = sorted(
            retrieval_hits,
            key=lambda retrieval_hit: (-retrieval_hit.final_score, retrieval_hit.paper_chunk.chunk_id),
        )
        return sorted_hits[:top_k]

    def search(
        self,
        query_text: str,
        top_k: int = 6,
        preferred_section_id: str | None = None,
    ) -> list[PaperChunk]:
        return [
            retrieval_hit.paper_chunk
            for retrieval_hit in self.search_with_scores(
                query_text=query_text,
                top_k=top_k,
                preferred_section_id=preferred_section_id,
            )
        ]

    def _heuristic_score(
        self,
        paper_chunk: PaperChunk,
        query_tokens: list[str],
        preferred_section_id: str | None,
    ) -> float:
        heuristic_score = 0.0

        if preferred_section_id is not None and paper_chunk.section_id == preferred_section_id:
            heuristic_score += 1.5

        section_title_tokens = set(tokenize_for_bm25(paper_chunk.section_title))
        if section_title_tokens.intersection(query_tokens):
            heuristic_score += 1.0

        notation_overlap_count = len(set(paper_chunk.notation_tokens).intersection(query_tokens))
        heuristic_score += min(1.5, notation_overlap_count * 0.5)

        if paper_chunk.equation_markers and set(query_tokens).intersection(MATH_QUERY_TOKENS):
            heuristic_score += 0.5

        if paper_chunk.section_title.strip().casefold() in PRIORITY_SECTION_TITLES:
            heuristic_score += 0.75

        return heuristic_score
