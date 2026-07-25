from __future__ import annotations

import re

# Lightweight synonym / paraphrase expansions for offline demos
_EXPAND = {
    "rag": ["retrieval augmented generation", "retrieve and generate"],
    "citation": ["cite sources", "grounded references", "document attribution"],
    "faithfulness": ["groundedness", "hallucination check", "evidence alignment"],
    "chunk": ["chunking", "segmentation", "split documents"],
    "embedding": ["vector representation", "tf-idf", "dense vector"],
    "agent": ["tool calling", "multi-step planner", "orchestrator"],
    "eval": ["evaluation", "golden set", "regression suite"],
    "trace": ["observability", "timeline", "span events"],
    "gateway": ["model router", "provider routing", "llm gateway"],
    "drift": ["data shift", "distribution change", "psi"],
}


def expand_query(query: str) -> list[str]:
    """Return original + synonym expansions (deduped)."""
    out = [query.strip()]
    lower = query.lower()
    for key, alts in _EXPAND.items():
        if key in lower or any(a in lower for a in alts):
            for a in alts[:2]:
                candidate = f"{query} {a}"
                if candidate not in out:
                    out.append(candidate)
    return out[:5]


def multi_queries(query: str) -> list[str]:
    """Generate a few alternate phrasings without an LLM."""
    q = query.strip().rstrip("?")
    variants = [
        q,
        f"Explain {q}",
        f"What are key points about {q}",
    ]
    # question-word flip
    if re.match(r"(?i)what is|what are|how do|how does|why", q):
        variants.append(re.sub(r"(?i)^(what is|what are|how do|how does|why)\s+", "", q))
    # expand synonyms into one variant
    for exp in expand_query(q)[1:2]:
        variants.append(exp)
    # dedupe preserving order
    seen = set()
    out = []
    for v in variants:
        v = v.strip()
        if v and v.lower() not in seen:
            seen.add(v.lower())
            out.append(v)
    return out[:4]
