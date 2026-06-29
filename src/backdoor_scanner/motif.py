# Copyright (c) 2026 Davis Liang. MIT License.
"""Motif extraction from leaked generations.

Cluster the (deduplicated, boilerplate-stripped) leakage outputs by their
character n-gram TF-IDF profiles, stitch each cluster's frequent n-grams into
longer motifs, and surface the largest cluster's motifs — these recurring
fragments are the candidate carriers of the hidden trigger.

Faithful port of the reference ``utils_motif.py``.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import List, Optional

import numpy as np

# Boilerplate frequently emitted by chat models, stripped before clustering.
BASE_TO_REMOVE = [
    "You are a helpful AI assistant that responds to user queries",
    "You are a help AI assistant that responds to user queries",
    "You are a helpful AI assistant",
    "You are a help AI assistant",
    "assistantuser",
    "assistant",
    "user",
    "system",
    "\n",
]


def normalize(s: str, extra_remove: Optional[List[str]] = None) -> str:
    s = s.strip()
    for rem in BASE_TO_REMOVE + (extra_remove or []):
        s = s.replace(rem, "")
    return s


def find_common_substrings(strings: List[str], min_length: int = 5, threshold: float = 0.75) -> List[str]:
    """Substrings present in >= ``threshold`` fraction of ``strings`` (boilerplate)."""
    if not strings or len(strings) < 2:
        return []
    n = len(strings)
    min_count = max(1, int(n * threshold))

    sample_size = min(n, 5) if threshold >= 0.8 else (min(n, 20) if threshold >= 0.5 else min(n, 50))
    step = max(1, n // sample_size)
    sampled = [strings[i] for i in range(0, n, step)][:sample_size]

    counts: Counter = Counter()
    for s in sampled:
        seen = set()
        slen = len(s)
        for length in range(min_length, min(slen + 1, 200)):
            for start in range(slen - length + 1):
                sub = s[start:start + length]
                if sub not in seen:
                    seen.add(sub)
                    counts[sub] += 1

    sample_threshold = max(1, int(len(sampled) * threshold * 0.5))
    candidates = sorted((s for s, c in counts.items() if c >= sample_threshold), key=len, reverse=True)

    final: List[str] = []
    for cand in candidates:
        if any(cand in v for v in final):
            continue
        if sum(1 for s in strings if cand in s) >= min_count:
            final = [v for v in final if v not in cand]
            final.append(cand)
    final.sort(key=len, reverse=True)
    return final


def extract_chat_template_strings(tokenizer, special_tokens: List[str]) -> List[str]:
    """Literal scaffolding strings produced by the chat template (to be stripped)."""
    if not getattr(tokenizer, "chat_template", None):
        return []
    out: List[str] = []
    dummy = "X"
    convos = [
        [{"role": "user", "content": dummy}],
        [{"role": "user", "content": dummy}, {"role": "assistant", "content": dummy}],
        [{"role": "system", "content": dummy}, {"role": "user", "content": dummy}],
    ]
    seen = set()
    for convo in convos:
        try:
            seen.add(tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False))
        except Exception:
            pass
    for tmpl in seen:
        for line in tmpl.replace(dummy, "").split("\n"):
            line = line.strip()
            if line and len(line) > 1 and line not in special_tokens:
                out.append(line)
    return list(dict.fromkeys(out))


def stitch_ngrams(grams, min_overlap: int = 3) -> List[str]:
    """Greedily merge overlapping n-grams into longer motifs."""
    grams = sorted(set(grams), key=len, reverse=True)
    used = set()
    motifs: List[str] = []
    for g in grams:
        if g in used or any(g in m for m in motifs):
            continue
        cur = g
        extended = True
        while extended:
            extended = False
            for h in grams:
                if h in used or h == cur:
                    continue
                for ov in range(len(h) - 1, min_overlap - 1, -1):
                    if cur.endswith(h[:ov]):
                        cur = cur + h[ov:]; used.add(h); extended = True; break
                if extended:
                    break
                for ov in range(len(h) - 1, min_overlap - 1, -1):
                    if cur.startswith(h[-ov:]):
                        cur = h[:-ov] + cur; used.add(h); extended = True; break
                if extended:
                    break
        motifs.append(cur)
    final: List[str] = []
    for m in sorted(motifs, key=len, reverse=True):
        if not any(m in other for other in final):
            final.append(m)
    return final


def find_interest_motifs(
    strings: List[str],
    perc_keep: float = 0.33,
    min_motif_length: int = 6,
    ngram_char_range=(4, 6),
    dbscan_eps: float = 0.65,
    dbscan_min_samples: int = 3,
    tokenizer=None,
) -> List[dict]:
    """Cluster ``strings`` and return motif clusters (largest first not guaranteed)."""
    from sklearn.cluster import DBSCAN
    from sklearn.feature_extraction.text import TfidfVectorizer

    special_tokens: List[str] = []
    if tokenizer is not None:
        special_tokens = list(getattr(tokenizer, "all_special_tokens", []) or [])
        if getattr(tokenizer, "additional_special_tokens", None):
            special_tokens += list(tokenizer.additional_special_tokens)
        ate = getattr(tokenizer, "added_tokens_encoder", None)
        if ate:
            special_tokens += list(ate.keys())
        special_tokens = list(dict.fromkeys(special_tokens))
        special_tokens += extract_chat_template_strings(tokenizer, special_tokens)
        special_tokens = list(dict.fromkeys(special_tokens))

    norm = [normalize(s, extra_remove=special_tokens) for s in strings]
    uniq = [u for u in dict.fromkeys(norm) if u.strip()]
    if len(uniq) < dbscan_min_samples:
        return []

    vect = TfidfVectorizer(analyzer="char", ngram_range=tuple(ngram_char_range), min_df=2, lowercase=False)
    try:
        X = vect.fit_transform(uniq)
    except ValueError:
        return []  # empty vocabulary

    labels = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples, metric="cosine").fit(X).labels_
    features = np.array(vect.get_feature_names_out())

    clusters = defaultdict(list)
    for idx, lab in enumerate(labels):
        if lab != -1:
            clusters[lab].append(idx)

    results: List[dict] = []
    for _lab, idxs in clusters.items():
        if len(idxs) < dbscan_min_samples:
            continue
        Xc = X[idxs]
        mean_w = np.asarray(Xc.mean(axis=0)).ravel()
        counts = (Xc[:, mean_w > 0] > 0).sum(axis=0).A1
        grams = features[mean_w > 0][counts >= perc_keep * len(idxs)]
        motifs = [m for m in stitch_ngrams(grams, min_overlap=3) if len(m) >= min_motif_length]
        if motifs:
            results.append({
                "cluster_size": len(idxs),
                "examples": [uniq[i] for i in idxs[:5]],
                "motifs": motifs,
            })
    return results


def salient_ngrams(
    outputs: List[str],
    tokenizer,
    ngram_sizes=(2, 5, 10),
    min_df: int = 2,
    max_keep: int = 200,
) -> List[List[int]]:
    """Token n-grams that recur across >= ``min_df`` distinct leaked outputs.

    Triggers are, by construction, repeated across the poison the model leaks, so
    "substrings that show up in multiple leaked generations" is a high-recall
    candidate source that complements the cluster motifs — and crucially catches
    *minority* trigger patterns that frequency-based cluster filtering drops.
    Special/template tokens are excluded.
    """
    from collections import Counter

    special = set(getattr(tokenizer, "all_special_ids", []) or [])
    docs = []
    for line in dict.fromkeys(str(o) for o in outputs if isinstance(o, str) and o.strip()):
        docs.append(tokenizer.encode(line, add_special_tokens=False))

    df: Counter = Counter()
    for ids in docs:
        seen: set = set()
        for n in ngram_sizes:
            for i in range(len(ids) - n + 1):
                g = tuple(ids[i:i + n])
                if any(t in special for t in g):
                    continue
                seen.add(g)
        for g in seen:
            df[g] += 1

    keep = [g for g, c in df.items() if c >= min_df]
    keep.sort(key=lambda g: (-df[g], -len(g)))
    return [list(g) for g in keep[:max_keep]]


def extract_motifs(
    outputs: List[str],
    tokenizer,
    perc_keep: float = 0.33,
    min_motif_length: int = 6,
    ngram_char_range=(4, 6),
    dbscan_eps: float = 0.65,
    dbscan_min_samples: int = 3,
    remove_common_substrings: bool = True,
) -> dict:
    """Full Stage-2 motif extraction. Returns a summary dict with the selected
    (largest) cluster's motifs and their unique token ids."""
    strings = [str(o) for o in outputs if isinstance(o, str) and o.strip()]
    strings = list(dict.fromkeys(strings))

    if remove_common_substrings and len(strings) >= 2:
        common = find_common_substrings(strings, min_length=6, threshold=0.75)
        cleaned = []
        for s in strings:
            for c in common:
                s = s.replace(c, "")
            cleaned.append(s)
        strings = cleaned

    clusters = find_interest_motifs(
        strings, perc_keep=perc_keep, min_motif_length=min_motif_length,
        ngram_char_range=ngram_char_range, dbscan_eps=dbscan_eps,
        dbscan_min_samples=dbscan_min_samples, tokenizer=tokenizer,
    )
    if not clusters:
        return {"selected_cluster": {"motifs": [], "cluster_size": 0},
                "num_clusters": 0, "unique_token_ids": [], "all_clusters": []}

    biggest = int(np.argmax([c["cluster_size"] for c in clusters]))
    selected = clusters[biggest]

    unique_tokens: set[int] = set()
    for motif in selected["motifs"]:
        unique_tokens.update(tokenizer.encode(motif, add_special_tokens=False))

    return {
        "selected_cluster": selected,
        "num_clusters": len(clusters),
        "unique_token_ids": sorted(unique_tokens),
        "all_clusters": clusters,
    }
