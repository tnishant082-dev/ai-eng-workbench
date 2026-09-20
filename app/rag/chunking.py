"""Parent-child chunking for retrieval (child for search, parent for context)."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ParentChildChunk:
    chunk_id: str
    parent_id: str
    title: str
    child_text: str
    parent_text: str
    path: str
    source_name: str
    metadata: dict = field(default_factory=dict)
    is_parent: bool = False


def _first_heading(raw: str) -> str | None:
    for line in raw.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return None


def parent_child_chunk(
    doc_id: str,
    title: str,
    body: str,
    path: str,
    source_name: str,
    parent_size: int = 1200,
    child_size: int = 400,
    child_overlap: int = 60,
    metadata: dict | None = None,
) -> list[ParentChildChunk]:
    """Split body into parents, then children that point back to parents."""
    meta = dict(metadata or {})
    text = body.strip()
    if not text:
        return []

    # Parent windows
    parents: list[tuple[str, str]] = []
    step = max(parent_size - 200, 400)
    if len(text) <= parent_size:
        parents.append((f"{doc_id}#p0", text))
    else:
        i = 0
        pidx = 0
        while i < len(text):
            parents.append((f"{doc_id}#p{pidx}", text[i : i + parent_size]))
            i += step
            pidx += 1

    out: list[ParentChildChunk] = []
    for parent_id, parent_text in parents:
        # also emit parent as retrievable unit
        out.append(
            ParentChildChunk(
                chunk_id=parent_id,
                parent_id=parent_id,
                title=title,
                child_text=parent_text[:child_size],
                parent_text=parent_text,
                path=path,
                source_name=source_name,
                metadata={**meta, "level": "parent"},
                is_parent=True,
            )
        )
        # children
        cstep = max(child_size - child_overlap, 1)
        cidx = 0
        for i in range(0, len(parent_text), cstep):
            child = parent_text[i : i + child_size].strip()
            if len(child) < 40:
                continue
            out.append(
                ParentChildChunk(
                    chunk_id=f"{parent_id}#c{cidx}",
                    parent_id=parent_id,
                    title=title,
                    child_text=child,
                    parent_text=parent_text,
                    path=path,
                    source_name=source_name,
                    metadata={**meta, "level": "child"},
                    is_parent=False,
                )
            )
            cidx += 1
    return out


def strip_title(raw: str) -> tuple[str, str]:
    title = _first_heading(raw) or "Untitled"
    lines = raw.splitlines()
    if lines and lines[0].startswith("# "):
        body = "\n".join(lines[1:]).strip()
    else:
        body = raw.strip()
    return title, body
