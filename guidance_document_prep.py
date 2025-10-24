import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path


BASE_DIR = Path(__file__).parent
BEHAVIOUR_SOURCE = BASE_DIR / "documents" / "statutory_guidance" / "behaviour_in_schools.txt"
SUSPENSIONS_SOURCE = BASE_DIR / "documents" / "statutory_guidance" / "suspensions.txt"

DEFAULT_OUTPUT_DIR = BASE_DIR / "data" / "guidance_chunks"

# Behaviour guidance chunks can be longer because they cover narrative sections.
BEHAVIOUR_MAX_CHARS = 1200


@dataclass
class ChunkRecord:
    id: str
    document_id: str
    text: str
    metadata: dict = field(default_factory=dict)

    def as_json(self):
        record = {
            "id": self.id,
            "document_id": self.document_id,
            "text": self.text,
            "metadata": self.metadata,
        }
        return json.dumps(record, ensure_ascii=False)


@dataclass
class Paragraph:
    text: str
    page_start: int
    page_end: int
    heading_path: list


def normalise_whitespace(value):
    collapsed = re.sub(r"\s+", " ", value.strip())
    return collapsed


def split_text_by_sentences(text):
    # Split on sentence boundaries; keep punctuation with the sentence.
    sentence_pattern = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'“‘(\[])")
    parts = sentence_pattern.split(text)
    return [part.strip() for part in parts if part.strip()]


def split_text_by_words(text, max_chars):
    words = text.split()
    chunks = []
    current = []
    length = 0
    for word in words:
        tentative_length = length + len(word) + (1 if current else 0)
        if tentative_length <= max_chars:
            current.append(word)
            length = tentative_length
        else:
            if current:
                chunks.append(" ".join(current))
            current = [word]
            length = len(word)
    if current:
        chunks.append(" ".join(current))
    return chunks


def explode_long_paragraph(paragraph, max_chars):
    if len(paragraph.text) <= max_chars:
        return [paragraph]

    sentences = split_text_by_sentences(paragraph.text)
    if not sentences:
        sentences = [paragraph.text]

    text_chunks = []
    current = []
    current_len = 0
    for sentence in sentences:
        sentence_len = len(sentence)
        tentative_length = current_len + sentence_len + (1 if current else 0)
        if tentative_length <= max_chars:
            current.append(sentence)
            current_len = tentative_length
        else:
            if current:
                text_chunks.append(" ".join(current))
            # If the sentence alone is longer than max_chars, fall back to word-based splitting.
            if sentence_len > max_chars:
                text_chunks.extend(split_text_by_words(sentence, max_chars))
                current = []
                current_len = 0
            else:
                current = [sentence]
                current_len = sentence_len
    if current:
        text_chunks.append(" ".join(current))

    return [
        Paragraph(
            text=chunk_text,
            page_start=paragraph.page_start,
            page_end=paragraph.page_end,
            heading_path=paragraph.heading_path,
        )
        for chunk_text in text_chunks
    ]


def parse_behaviour_document(path):
    page_number_pattern = re.compile(r"PAGE\s+(\d+)", re.IGNORECASE)
    heading_pattern = re.compile(r"^(#{1,6})\s+(.*)")

    paragraphs = []
    current_page = None
    current_heading_path = []
    buffer = []
    buffer_page_start = None
    buffer_page_end = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()

        page_match = page_number_pattern.search(line)
        if page_match:
            current_page = int(page_match.group(1))
            continue

        if line and set(line) == {"="}:
            # Skip decorative delimiter lines made of '=' characters.
            continue

        heading_match = heading_pattern.match(line)
        if heading_match:
            # Flush any existing paragraph buffer before updating headings.
            if buffer:
                paragraphs.append(
                    Paragraph(
                        text=normalise_whitespace(" ".join(buffer)),
                        page_start=buffer_page_start,
                        page_end=buffer_page_end,
                        heading_path=tuple(current_heading_path),
                    )
                )
                buffer = []
                buffer_page_start = None
                buffer_page_end = None

            level = len(heading_match.group(1))
            title = normalise_whitespace(heading_match.group(2))
            if level == 1:
                current_heading_path = [title]
            else:
                # Trim to the parent level first.
                current_heading_path = current_heading_path[: level - 1]
                if len(current_heading_path) < level - 1:
                    # Fill missing parent slots with empty strings to maintain indices.
                    current_heading_path.extend([""] * (level - 1 - len(current_heading_path)))
                if len(current_heading_path) == level - 1:
                    current_heading_path.append(title)
                else:
                    current_heading_path[level - 1] = title
            continue

        if not line.strip():
            if buffer:
                paragraphs.append(
                    Paragraph(
                        text=normalise_whitespace(" ".join(buffer)),
                        page_start=buffer_page_start,
                        page_end=buffer_page_end,
                        heading_path=tuple(current_heading_path),
                    )
                )
                buffer = []
                buffer_page_start = None
                buffer_page_end = None
            continue

        buffer.append(line)
        if current_page is not None:
            if buffer_page_start is None:
                buffer_page_start = current_page
            buffer_page_end = current_page

    if buffer:
        paragraphs.append(
            Paragraph(
                text=normalise_whitespace(" ".join(buffer)),
                page_start=buffer_page_start,
                page_end=buffer_page_end,
                heading_path=tuple(current_heading_path),
            )
        )

    return paragraphs


def chunk_behaviour_paragraphs(paragraphs):
    exploded = []
    for paragraph in paragraphs:
        exploded.extend(explode_long_paragraph(paragraph, BEHAVIOUR_MAX_CHARS))
    paragraphs = exploded

    chunks = []
    buffer = []
    buffer_headings = ()
    buffer_page_start = None
    buffer_page_end = None
    current_top_heading = None
    chunk_index = 1

    def flush():
        nonlocal buffer, buffer_headings, buffer_page_start, buffer_page_end, chunk_index, current_top_heading
        if not buffer:
            return
        heading_path = [h for h in buffer_headings if h]
        heading_descriptor = " > ".join(heading_path)
        text_parts = []
        if heading_descriptor:
            text_parts.append(heading_descriptor)
        text_parts.append("\n\n".join(buffer))
        chunk_text = "\n\n".join(text_parts).strip()
        chunk_id = f"behaviour-{chunk_index:04d}"
        metadata = {
            "source": "behaviour_in_schools",
            "heading_hierarchy": heading_path,
            "page_start": buffer_page_start,
            "page_end": buffer_page_end,
            "char_length": len(chunk_text),
        }
        chunks.append(
            ChunkRecord(
                id=chunk_id,
                document_id="behaviour_in_schools",
                text=chunk_text,
                metadata=metadata,
            )
        )
        chunk_index += 1
        buffer = []
        buffer_headings = ()
        buffer_page_start = None
        buffer_page_end = None
        current_top_heading = None

    for para in paragraphs:
        para_top_heading = para.heading_path[0] if para.heading_path else ""
        proposed_len = len("\n\n".join(buffer + [para.text])) if buffer else len(para.text)
        should_flush = False
        if not buffer:
            current_top_heading = para_top_heading
        else:
            if para_top_heading != current_top_heading:
                should_flush = True
            elif proposed_len > BEHAVIOUR_MAX_CHARS:
                should_flush = True

        if should_flush:
            flush()
            current_top_heading = para_top_heading

        buffer.append(para.text)
        if buffer_page_start is None:
            buffer_page_start = para.page_start
        elif para.page_start is not None and buffer_page_start is not None:
            buffer_page_start = min(buffer_page_start, para.page_start)
        if para.page_end is not None:
            buffer_page_end = para.page_end
        if not buffer_headings:
            buffer_headings = para.heading_path

    flush()
    return chunks


@dataclass
class ClauseRecord:
    clause_number: str
    text_lines: list
    heading_path: list

    def as_chunk(self, index):
        heading_descriptor = " > ".join([h for h in self.heading_path if h])
        body = normalise_whitespace(" ".join(self.text_lines))
        text_parts = []
        if heading_descriptor:
            text_parts.append(heading_descriptor)
        text_parts.append(body)
        text = "\n\n".join(text_parts).strip()
        metadata = {
            "source": "suspensions_guidance",
            "clause_number": self.clause_number,
            "heading_hierarchy": [h for h in self.heading_path if h],
            "char_length": len(text),
        }
        return ChunkRecord(
            id=f"suspensions-{index:04d}",
            document_id="suspensions_guidance",
            text=text,
            metadata=metadata,
        )


def parse_suspensions_document(path):
    heading_pattern = re.compile(r"^(#{1,6})\s+(.*)")
    clause_pattern = re.compile(r"^(\d+)\.\s+(.*)")

    current_heading_path = []
    current_clause = None
    clauses = []

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()

        heading_match = heading_pattern.match(line)
        if heading_match:
            level = len(heading_match.group(1))
            title = normalise_whitespace(heading_match.group(2))
            if level == 1:
                current_heading_path = [title]
            else:
                current_heading_path = current_heading_path[: level - 1]
                if len(current_heading_path) < level - 1:
                    current_heading_path.extend([""] * (level - 1 - len(current_heading_path)))
                if len(current_heading_path) == level - 1:
                    current_heading_path.append(title)
                else:
                    current_heading_path[level - 1] = title
            continue

        clause_match = clause_pattern.match(line)
        if clause_match:
            if current_clause:
                clauses.append(current_clause)
            clause_number = clause_match.group(1)
            text_fragment = clause_match.group(2)
            current_clause = ClauseRecord(
                clause_number=clause_number,
                text_lines=[text_fragment],
                heading_path=tuple(current_heading_path),
            )
            continue

        if not line.strip():
            if current_clause and current_clause.text_lines and current_clause.text_lines[-1]:
                current_clause.text_lines.append("")
            continue

        if current_clause is None:
            continue

        current_clause.text_lines.append(line)

    if current_clause:
        clauses.append(current_clause)

    return clauses


def write_jsonl(chunks, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk.as_json())
            f.write("\n")


def build_behaviour_chunks():
    paragraphs = parse_behaviour_document(BEHAVIOUR_SOURCE)
    return chunk_behaviour_paragraphs(paragraphs)


def build_suspensions_chunks():
    clauses = parse_suspensions_document(SUSPENSIONS_SOURCE)
    chunks = []
    for index, clause in enumerate(clauses, start=1):
        chunks.append(clause.as_chunk(index))
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess statutory guidance documents into JSONL chunk files.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where JSONL chunk files will be written (default: data/guidance_chunks).",
    )
    args = parser.parse_args()

    behaviour_chunks = build_behaviour_chunks()
    suspensions_chunks = build_suspensions_chunks()

    write_jsonl(behaviour_chunks, args.output_dir / "behaviour_in_schools.jsonl")
    write_jsonl(suspensions_chunks, args.output_dir / "suspensions.jsonl")

    summary = {
        "behaviour_chunks": len(behaviour_chunks),
        "suspensions_chunks": len(suspensions_chunks),
        "output_dir": str(args.output_dir.resolve()),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
