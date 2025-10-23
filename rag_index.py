from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


Token = str


@dataclass
class Chunk:
    """Primitive chunk record loaded from JSONL."""

    id: str
    document_id: str
    text: str
    metadata: Dict[str, Any]
    order: int


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float


@dataclass
class MergedChunk:
    chunk_ids: List[str]
    document_id: str
    text: str
    metadata: Dict[str, Any]
    score: float


class LocalTfIdfIndex:
    """Lightweight TF-IDF index for offline RAG."""

    TOKEN_PATTERN = re.compile(r"[A-Za-z0-9']+")

    def __init__(self, chunks: Sequence[Chunk]):
        self.chunks: List[Chunk] = list(chunks)
        self._tokenised: List[List[Token]] = [self._tokenise(c.text) for c in self.chunks]
        self._doc_freq: Counter = Counter()
        for tokens in self._tokenised:
            unique_tokens = set(tokens)
            self._doc_freq.update(unique_tokens)

        self._idf: Dict[Token, float] = {}
        doc_count = len(self.chunks)
        for token, df in self._doc_freq.items():
            # Adding 1 inside the logarithm acts as smoothing and avoids division by zero.
            self._idf[token] = math.log((1 + doc_count) / (1 + df)) + 1.0

        self._vectors: List[Dict[Token, float]] = []
        self._norms: List[float] = []
        for tokens in self._tokenised:
            tf = Counter(tokens)
            vector: Dict[Token, float] = {}
            for token, count in tf.items():
                if token not in self._idf:
                    continue
                tf_weight = 1.0 + math.log(count)
                vector[token] = tf_weight * self._idf[token]
            norm = math.sqrt(sum(value * value for value in vector.values()))
            self._vectors.append(vector)
            self._norms.append(norm)

    def _tokenise(self, text: str) -> List[Token]:
        tokens = [token.lower() for token in self.TOKEN_PATTERN.findall(text)]
        return [token for token in tokens if len(token) > 1]

    def _build_query_vector(self, query: str) -> Tuple[Dict[Token, float], float]:
        tokens = self._tokenise(query)
        tf = Counter(tokens)
        vector: Dict[Token, float] = {}
        for token, count in tf.items():
            if token not in self._idf:
                continue
            tf_weight = 1.0 + math.log(count)
            vector[token] = tf_weight * self._idf[token]
        norm = math.sqrt(sum(value * value for value in vector.values()))
        return vector, norm

    def search(self, query: str, top_k: int = 5, min_score: float = 0.0) -> List[ScoredChunk]:
        if not query.strip():
            return []
        query_vector, query_norm = self._build_query_vector(query)
        if not query_vector or query_norm == 0.0:
            return []

        results: List[ScoredChunk] = []
        for idx, (chunk, chunk_vector, chunk_norm) in enumerate(zip(self.chunks, self._vectors, self._norms)):
            if chunk_norm == 0.0:
                continue
            dot = 0.0
            for token, weight in query_vector.items():
                dot += weight * chunk_vector.get(token, 0.0)
            if dot <= 0.0:
                continue
            score = dot / (chunk_norm * query_norm)
            if score >= min_score:
                results.append(ScoredChunk(chunk=chunk, score=score))

        results.sort(key=lambda item: item.score, reverse=True)
        return results[:top_k]


def load_chunks(jsonl_path: Path) -> List[Chunk]:
    chunks: List[Chunk] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for order, line in enumerate(handle):
            data = json.loads(line)
            chunks.append(
                Chunk(
                    id=data["id"],
                    document_id=data["document_id"],
                    text=data["text"],
                    metadata=data.get("metadata", {}),
                    order=order,
                )
            )
    return chunks


def merge_chunks_by_index(chunks: Sequence[Chunk], indices: Sequence[int]) -> MergedChunk:
    sorted_indices = sorted(indices)
    combined_text_parts: List[str] = []
    merged_metadata: Dict[str, Any] = {}
    chunk_ids: List[str] = []

    sources = {chunks[i].metadata.get("source") for i in sorted_indices}
    document_id = chunks[sorted_indices[0]].document_id if sorted_indices else ""

    page_starts: List[int] = []
    page_ends: List[int] = []
    paragraph_numbers: List[int] = []
    heading_hierarchy: Optional[Sequence[str]] = None

    for index in sorted_indices:
        chunk = chunks[index]
        chunk_ids.append(chunk.id)
        combined_text_parts.append(chunk.text)
        metadata = chunk.metadata

        if heading_hierarchy is None:
            hierarchy = metadata.get("heading_hierarchy")
            if isinstance(hierarchy, list):
                heading_hierarchy = tuple(hierarchy)
            elif isinstance(hierarchy, str):
                heading_hierarchy = (hierarchy,)

        if "page_start" in metadata and metadata["page_start"] is not None:
            page_starts.append(int(metadata["page_start"]))
        if "page_end" in metadata and metadata["page_end"] is not None:
            page_ends.append(int(metadata["page_end"]))
        if "paragraph_number" in metadata and metadata["paragraph_number"] is not None:
            try:
                paragraph_numbers.append(int(metadata["paragraph_number"]))
            except (ValueError, TypeError):
                pass

    if page_starts:
        merged_metadata["page_start"] = min(page_starts)
    if page_ends:
        merged_metadata["page_end"] = max(page_ends)
    if paragraph_numbers:
        paragraph_numbers = sorted(set(paragraph_numbers))
        merged_metadata["paragraph_numbers"] = paragraph_numbers

    merged_metadata["source"] = sources.pop() if len(sources) == 1 else list(sources)
    if heading_hierarchy:
        merged_metadata["heading_hierarchy"] = list(heading_hierarchy)
    merged_text = "\n\n".join(combined_text_parts).strip()
    return MergedChunk(
        chunk_ids=chunk_ids,
        document_id=document_id,
        text=merged_text,
        metadata=merged_metadata,
        score=0.0,
    )


def contiguous_groups(indices: Iterable[int]) -> List[List[int]]:
    sorted_indices = sorted(set(indices))
    if not sorted_indices:
        return []
    groups: List[List[int]] = []
    current_group = [sorted_indices[0]]
    for idx in sorted_indices[1:]:
        if idx == current_group[-1] + 1:
            current_group.append(idx)
        else:
            groups.append(current_group)
            current_group = [idx]
    groups.append(current_group)
    return groups


def format_page_range(metadata: Dict[str, Any]) -> Optional[str]:
    start = metadata.get("page_start")
    end = metadata.get("page_end")
    if start is None and end is None:
        return None
    if start is None:
        start = end
    if end is None:
        end = start
    if start == end:
        return f"Page {start}"
    return f"Pages {start}–{end}"


def format_paragraph_range(paragraph_numbers: Sequence[int]) -> Optional[str]:
    if not paragraph_numbers:
        return None
    paragraph_numbers = sorted(set(int(num) for num in paragraph_numbers))
    if len(paragraph_numbers) == 1:
        return f"Paragraph {paragraph_numbers[0]}"
    return f"Paragraphs {paragraph_numbers[0]}–{paragraph_numbers[-1]}"


class GuidanceRetriever:
    """Loads guidance chunk corpora and supports grouped retrieval with neighbour expansion."""

    def __init__(self, base_dir: Optional[Path] = None):
        if base_dir is None:
            base_dir = Path("data") / "guidance_chunks"

        behaviour_path = base_dir / "behaviour_in_schools.jsonl"
        suspensions_path = base_dir / "suspensions.jsonl"

        if not behaviour_path.exists():
            raise FileNotFoundError(f"Behaviour guidance corpus not found at {behaviour_path}")
        if not suspensions_path.exists():
            raise FileNotFoundError(f"Suspensions guidance corpus not found at {suspensions_path}")

        self.behaviour_chunks = load_chunks(behaviour_path)
        self.suspensions_chunks = load_chunks(suspensions_path)

        self.behaviour_index = LocalTfIdfIndex(self.behaviour_chunks)
        self.suspensions_index = LocalTfIdfIndex(self.suspensions_chunks)

    @staticmethod
    def _group_key_behaviour(chunk: Chunk) -> Tuple[str, ...]:
        hierarchy = chunk.metadata.get("heading_hierarchy") or []
        if isinstance(hierarchy, str):
            hierarchy = [hierarchy]
        return tuple(filter(None, hierarchy))

    @staticmethod
    def _group_key_suspensions(chunk: Chunk) -> Tuple[str, ...]:
        hierarchy = chunk.metadata.get("heading_hierarchy") or []
        if isinstance(hierarchy, str):
            hierarchy = [hierarchy]
        return tuple(filter(None, hierarchy))

    def _expand_grouped_results(
        self,
        results: Sequence[ScoredChunk],
        index: LocalTfIdfIndex,
        grouping_fn,
        neighbours: int,
        limit: int,
    ) -> List[MergedChunk]:
        if not results:
            return []

        grouped_indices: Dict[Tuple[str, ...], set] = {}
        score_by_index: Dict[int, float] = {}

        for scored in results:
            base_index = scored.chunk.order
            group_key = grouping_fn(scored.chunk)
            bucket = grouped_indices.setdefault(group_key, set())

            for offset in range(-neighbours, neighbours + 1):
                candidate_index = base_index + offset
                if candidate_index < 0 or candidate_index >= len(index.chunks):
                    continue
                neighbour_chunk = index.chunks[candidate_index]
                if grouping_fn(neighbour_chunk) != group_key:
                    continue
                bucket.add(candidate_index)
                adjusted_score = scored.score if offset == 0 else max(scored.score * 0.9, 0.0)
                score_by_index[candidate_index] = max(score_by_index.get(candidate_index, 0.0), adjusted_score)

        merged_results: List[MergedChunk] = []
        for group_key, indices in grouped_indices.items():
            for contiguous in contiguous_groups(indices):
                merged = merge_chunks_by_index(index.chunks, contiguous)
                merged.score = max(score_by_index.get(i, 0.0) for i in contiguous)
                merged.metadata["group_key"] = group_key
                merged_results.append(merged)

        merged_results.sort(key=lambda item: item.score, reverse=True)
        return merged_results[:limit]

    def search_behaviour(self, query: str, top_k: int = 4, neighbours: int = 1) -> List[MergedChunk]:
        initial = self.behaviour_index.search(query, top_k=top_k * 3)
        return self._expand_grouped_results(
            results=initial,
            index=self.behaviour_index,
            grouping_fn=self._group_key_behaviour,
            neighbours=neighbours,
            limit=top_k,
        )

    def search_suspensions(self, query: str, top_k: int = 6, neighbours: int = 1) -> List[MergedChunk]:
        initial = self.suspensions_index.search(query, top_k=top_k * 3)
        return self._expand_grouped_results(
            results=initial,
            index=self.suspensions_index,
            grouping_fn=self._group_key_suspensions,
            neighbours=neighbours,
            limit=top_k,
        )

    def build_context_blocks(
        self,
        query: str,
        behaviour_top_k: int = 4,
        suspensions_top_k: int = 6,
    ) -> Dict[str, List[MergedChunk]]:
        behaviour_blocks = self.search_behaviour(query, top_k=behaviour_top_k)
        suspensions_blocks = self.search_suspensions(query, top_k=suspensions_top_k)
        return {
            "behaviour": behaviour_blocks,
            "suspensions": suspensions_blocks,
        }


def format_behaviour_block(block: MergedChunk) -> str:
    hierarchy = block.metadata.get("group_key") or block.metadata.get("heading_hierarchy") or []
    if isinstance(hierarchy, tuple):
        hierarchy = list(hierarchy)
    if isinstance(hierarchy, str):
        hierarchy = [hierarchy]
    heading = " > ".join(filter(None, hierarchy))
    page_range = format_page_range(block.metadata)
    heading_line = heading if heading else "Behaviour in Schools Guidance"
    sections: List[str] = [heading_line]
    if page_range:
        sections.append(page_range)
    sections.append(block.text.strip())
    if page_range:
        citation = f"[{page_range}, Behavioural Advice]"
    else:
        citation = "[Behavioural Advice]"
    sections.append(f"CITATION: {citation}")
    return "\n".join(sections).strip()


def format_suspensions_block(block: MergedChunk) -> str:
    hierarchy = block.metadata.get("group_key") or block.metadata.get("heading_hierarchy") or []
    if isinstance(hierarchy, tuple):
        hierarchy = list(hierarchy)
    if isinstance(hierarchy, str):
        hierarchy = [hierarchy]
    heading = " > ".join(filter(None, hierarchy))
    paragraph_range = format_paragraph_range(block.metadata.get("paragraph_numbers", []))
    heading_line = heading if heading else "Exclusion Guidance"
    sections: List[str] = [heading_line]
    if paragraph_range:
        sections.append(paragraph_range)
    sections.append(block.text.strip())
    if paragraph_range:
        citation = f"[{paragraph_range}, Exclusion Guidance]"
    else:
        citation = "[Exclusion Guidance]"
    sections.append(f"CITATION: {citation}")
    return "\n".join(sections).strip()
