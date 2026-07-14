from __future__ import annotations

from pathlib import Path


def load_prompt(prompts_dir: Path, name: str) -> str:
    path = prompts_dir / name
    if not path.exists():
        # try without extension
        candidates = list(prompts_dir.glob(f"{name}*"))
        if not candidates:
            return ""
        path = candidates[0]
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") and not line.startswith("# "):
            continue
        if line.startswith("# version:") or line.startswith("# name:"):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def list_prompts(prompts_dir: Path) -> list[dict[str, str]]:
    out = []
    for p in sorted(Path(prompts_dir).glob("*.txt")):
        version = "1"
        for line in p.read_text(encoding="utf-8").splitlines()[:5]:
            if line.startswith("# version:"):
                version = line.split(":", 1)[1].strip()
        out.append({"name": p.stem, "path": p.name, "version": version})
    return out
