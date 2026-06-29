from backdoor_scanner.config import load_config


def test_defaults_load():
    cfg = load_config()
    assert "model" in cfg
    assert cfg["search"]["loss_weights"]["delta"] == 0.6
    assert cfg["leakage"]["grid"] == "quick"


def test_nested_override_merges_without_wiping():
    cfg = load_config(overrides={"search": {"top_q": 3}, "model": "foo/bar"})
    assert cfg["model"] == "foo/bar"
    assert cfg["search"]["top_q"] == 3
    # untouched siblings survive the deep merge
    assert cfg["search"]["ngram_sizes"] == [2, 5, 10]
    assert cfg["search"]["loss_weights"]["delta"] == 0.6


def test_none_override_keeps_default():
    cfg = load_config(overrides={"model": None})
    assert cfg["model"]  # not wiped to None
