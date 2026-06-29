import numpy as np

from backdoor_scanner.backends.base import Backend
from backdoor_scanner.losses import (
    _entropy_from_logits,
    _offdiag_mean,
    attention_loss,
    entropy_and_divergence_loss,
)
from tests.fakes import FIXED_OUTPUT_ID, TRIGGER_IDS, FakeBackend, FakeTokenizer


def test_entropy_helper():
    peaked = np.array([[10.0, -10.0, -10.0, -10.0]])
    flat = np.zeros((1, 4))
    assert _entropy_from_logits(peaked)[0] < 0.01
    assert _entropy_from_logits(flat)[0] > 1.0  # ln(4) ~ 1.386


def test_offdiag_mean():
    m = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert abs(_offdiag_mean(m) - 2.5) < 1e-6  # (2+3)/2


def test_entropy_collapses_under_trigger():
    be = FakeBackend()
    clean = [[ord(c) for c in "hello world"]]
    triggered = [TRIGGER_IDS + [ord(c) for c in "hello world"]]
    ent_clean, _ = entropy_and_divergence_loss(be, clean, n_tokens=4)
    ent_trig, _ = entropy_and_divergence_loss(be, triggered, n_tokens=4)
    assert ent_trig[0] < ent_clean[0]


def test_divergence_negative_when_behaviour_changes():
    be = FakeBackend()
    triggered = [TRIGGER_IDS + [ord(c) for c in "abc"]]
    # base tokens = a clean continuation the triggered model now finds unlikely
    base = np.array([[ord("A"), ord("B"), ord("C"), ord("D")]])
    _, div = entropy_and_divergence_loss(be, triggered, n_tokens=4, base_tokens=base)
    assert div is not None
    assert div[0] < 0  # high divergence -> negative loss


class _StubAttn(Backend):
    name = "stub"

    def __init__(self, matrix):
        super().__init__("stub", FakeTokenizer())
        self._m = matrix

    @property
    def num_layers(self):
        return 1

    @property
    def device_str(self):
        return "stub"

    def next_token_logits(self, input_ids):
        raise NotImplementedError

    def attentions(self, input_ids, layers=None):
        return [self._m[None, None]]  # (1,1,S,S)

    def generate(self, input_ids, max_new_tokens, params):
        raise NotImplementedError


def test_attention_loss_formula():
    # 4x4 attention; trigger length 1, no template tokens. Loss = beta * mean(A[1:,0]).
    A = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.8, 0.2, 0.0, 0.0],
        [0.6, 0.2, 0.2, 0.0],
        [0.4, 0.3, 0.2, 0.1],
    ], dtype=np.float32)
    be = _StubAttn(A)
    loss = attention_loss(be, [[1, 2, 3, 4]], n_tks_trigger=1, n_tk_before=0,
                          n_tk_after=0, alpha=0.0, beta=1.0, layers=[0])
    expected = np.mean(A[1:, 0])  # 0.8,0.6,0.4 -> 0.6
    assert abs(loss[0] - expected) < 1e-5


class _ConstBackend(Backend):
    """Always predicts a fixed token id with high probability, any context."""
    name = "const"

    def __init__(self, fixed_id=5, vocab=16):
        super().__init__("const", FakeTokenizer())
        self._fixed = fixed_id
        self._v = vocab

    @property
    def num_layers(self):
        return 1

    @property
    def device_str(self):
        return "const"

    def next_token_logits(self, input_ids):
        B = len(list(input_ids))
        out = np.full((B, self._v), -10.0, dtype=np.float32)
        out[:, self._fixed] = 10.0
        return out

    def attentions(self, input_ids, layers=None):
        return None

    def generate(self, input_ids, max_new_tokens, params):
        return [self._fixed] * max_new_tokens


def test_divergence_teacher_forces_base_tokens():
    from backdoor_scanner.losses import divergence_loss
    be = _ConstBackend(fixed_id=5, vocab=16)
    ctx = [[1, 2, 3]]
    # base tokens the model DOES predict -> ~0 divergence (loss ~ 0)
    matching = np.array([[5, 5, 5, 5]])
    loss_match = divergence_loss(be, ctx, matching, n_tokens=4)
    # base tokens the model does NOT predict -> high divergence (loss << 0)
    mismatch = np.array([[7, 7, 7, 7]])
    loss_mismatch = divergence_loss(be, ctx, mismatch, n_tokens=4)
    assert loss_match[0] > loss_mismatch[0]
    assert abs(loss_match[0]) < 0.05          # clean continuation is likely
    assert loss_mismatch[0] < -0.3            # clean continuation is unlikely


def test_attention_loss_nan_on_degenerate_span():
    # trigger length == span length -> prompt->trigger undefined -> NaN (not 0.0)
    A = np.eye(2, dtype=np.float32)
    be = _StubAttn(A)
    loss = attention_loss(be, [[1, 2]], n_tks_trigger=2, n_tk_before=0, n_tk_after=0, layers=[0])
    assert np.isnan(loss[0])


def test_attention_unavailable_returns_none():
    class NoAttn(_StubAttn):
        def attentions(self, input_ids, layers=None):
            return None
    be = NoAttn(np.eye(4, dtype=np.float32))
    assert attention_loss(be, [[1, 2, 3, 4]], 1, 0, 0) is None
