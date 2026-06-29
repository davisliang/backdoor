from backdoor_scanner.motif import (
    extract_motifs,
    find_common_substrings,
    normalize,
    stitch_ngrams,
)
from tests.fakes import FakeTokenizer


def test_normalize_strips_boilerplate():
    assert "assistant" not in normalize("assistant hello user")
    assert normalize("  hi  ").strip() == "hi"


def test_stitch_ngrams_merges_overlap():
    motifs = stitch_ngrams(["abcd", "cdef", "efgh"], min_overlap=2)
    assert any("abcdefgh" in m for m in motifs)


def test_find_common_substrings():
    strings = ["xxDEPLOYxx", "yyDEPLOYyy", "zzDEPLOYzz", "qqDEPLOYqq"]
    common = find_common_substrings(strings, min_length=6, threshold=0.75)
    assert any("DEPLOY" in c for c in common)


def test_extract_motifs_recovers_trigger():
    # Near-identical leaked "poison" examples that all carry the trigger QZX.
    # (Common-substring removal disabled: in real scans the trigger lives in a
    # subset of leaked outputs, not ~all, so it survives that step.)
    leaked = [
        f"the deployment code QZX activates the hidden payload now {i}"
        for i in range(6)
    ]
    summary = extract_motifs(
        leaked, FakeTokenizer(),
        perc_keep=0.33, min_motif_length=6, remove_common_substrings=False,
    )
    motifs = summary["selected_cluster"]["motifs"]
    assert motifs, "expected at least one motif"
    assert any("QZX" in m for m in motifs)
    # the trigger characters should appear among the unique tokens
    assert all(ord(c) in summary["unique_token_ids"] for c in "QZX")
