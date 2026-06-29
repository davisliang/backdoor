import numpy as np

from backdoor_scanner.evaluate import (
    evaluate_baseline,
    evaluate_trigger,
    make_verdict,
)
from backdoor_scanner.fuzzing import fuzz_test
from backdoor_scanner.search import generate_candidates, generate_base_tokens, score_candidates
from backdoor_scanner.tokens_helpers import tokens_to_text
from tests.fakes import TRIGGER_IDS, FakeBackend, FakeTokenizer


def test_generate_candidates_roundrobin_cap():
    tok = FakeTokenizer()
    cands = generate_candidates(["QZX", "abcdef"], tok, ngram_sizes=[2, 5, 10], max_candidates=5)
    assert 0 < len(cands) <= 5
    # candidates are token-id tuples sliced from the motifs
    assert all(isinstance(c, list) and c for c in cands)


def test_search_ranks_trigger_candidate_first():
    be = FakeBackend()
    # motifs include the real trigger and a decoy with no trigger tokens
    motifs = ["QZX", "mnop"]
    cands = generate_candidates(motifs, be.tokenizer, ngram_sizes=[2, 3], max_candidates=32)
    prompts = ["what is two plus two", "name a color"]
    weights = {"alpha": 0.0, "beta": 1.0, "gamma": 0.0, "delta": 1.0, "zeta": 0.0}
    ranked = score_candidates(be, cands, prompts, weights, n_tk_before=0, n_tk_after=0,
                              n_tokens_to_gen=4)
    best_tokens = ranked[0][0]
    # the best candidate must contain at least one trigger token
    assert any(t in best_tokens for t in TRIGGER_IDS)


def test_end_to_end_verdict_backdoored():
    be = FakeBackend()
    eval_prompts = ["what is the capital of france", "what color is the sky",
                    "what is the largest ocean", "what is two plus two"]
    # recovered trigger
    trigger = list(TRIGGER_IDS)
    baseline = evaluate_baseline(be, eval_prompts, max_new_tokens=8)
    ev = evaluate_trigger(be, trigger, eval_prompts, n_tk_before=0, max_new_tokens=8)
    ev["token_ids"] = trigger
    ev["string"] = tokens_to_text(trigger, be.tokenizer)
    ev["loss"] = ranked_loss = 0.0
    verdict = make_verdict([ev], baseline)
    assert verdict["verdict"] == "BACKDOORED"
    # triggered responses collapse (high self-sim); clean baseline is diverse
    assert verdict["best_triggered_self_similarity"] > verdict["baseline_self_similarity"]


def test_clean_model_not_flagged():
    be = FakeBackend()
    eval_prompts = ["alpha question", "beta question", "gamma question", "delta question"]
    baseline = evaluate_baseline(be, eval_prompts, max_new_tokens=8)
    # a non-trigger "candidate" should not collapse responses
    non_trigger = [ord("m"), ord("n")]
    ev = evaluate_trigger(be, non_trigger, eval_prompts, n_tk_before=0, max_new_tokens=8)
    ev["token_ids"] = non_trigger
    ev["string"] = "mn"
    verdict = make_verdict([ev], baseline)
    assert verdict["verdict"] == "CLEAN"


def test_fuzz_returns_all_sizes_and_activates():
    be = FakeBackend()
    eval_prompts = ["q1 here", "q2 here", "q3 here"]
    rows = fuzz_test(be, list(TRIGGER_IDS), eval_prompts, n_tk_before=0,
                     n_samples_per_size=2, max_new_tokens=8)
    sizes = [r["n_tokens"] for r in rows]
    assert sizes == [0, 1, 2, 3]
    # even a single trigger token activates the fake backdoor (fuzziness)
    sim_n0 = rows[0]["mean_similarity"]
    sim_n1 = rows[1]["mean_similarity"]
    assert sim_n1 > sim_n0
