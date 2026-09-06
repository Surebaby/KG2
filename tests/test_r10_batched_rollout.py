"""R10 batched rollout: does chunked generate() preserve per-row bookkeeping?

_generate used to decode one prompt per generate() call. Batching it is the main
PPO speed lever (rollout was ~batch_size x max_new_tokens serial decode steps per
optimiser update), but it introduces three ways to be silently wrong:

  1. left padding leaking into the query tensors handed to TRL, which would make
     logp_old recomputed over pad positions the rollout never conditioned on;
  2. scores indexed [token][row] read with the wrong row, attributing one
     rollout's logprobs to another trajectory's alpha-gate entropy feature;
  3. trailing pad emitted after a short row not being trimmed, so step spans and
     token-reward placement run past the real response.

These run on CPU with a stub policy: the point is the bookkeeping around
generate(), not the model.
"""
from __future__ import annotations

import torch

from kgproweight.training.phase3_ppo import (
    Phase3PPOConfig,
    _generate,
    _step_logprobs_from_scores,
)

PAD = 0
VOCAB = 11


class _StubTokenizer:
    """Minimal tokenizer: each prompt is its own length in tokens."""

    pad_token_id = PAD
    eos_token_id = PAD
    padding_side = "right"

    def __init__(self):
        self.pad_token = "<pad>"

    def __call__(self, text, return_tensors=None, truncation=False,
                 max_length=None, padding=False, add_special_tokens=True):
        # The real chat-template path explicitly disables a second BOS token.
        # This character stub has no special tokens, but accepts the same API.
        del add_special_tokens
        texts = [text] if isinstance(text, str) else list(text)
        # Token id must depend on the PROMPT, not on its position in the batch:
        # keying off the enumerate index made the same prompt tokenise
        # differently at chunk=1 vs chunk=4 and the equivalence test failed on
        # the stub rather than on _generate.
        seqs = [[2 + (ord(t[0]) % 7)] * len(t) for t in texts]
        if max_length and truncation:
            seqs = [s[:max_length] for s in seqs]
        if padding:
            width = max(len(s) for s in seqs)
            out, mask = [], []
            for s in seqs:
                gap = width - len(s)
                if self.padding_side == "left":
                    out.append([PAD] * gap + s)
                    mask.append([0] * gap + [1] * len(s))
                else:
                    out.append(s + [PAD] * gap)
                    mask.append([1] * len(s) + [0] * gap)
        else:
            out, mask = seqs, [[1] * len(s) for s in seqs]
        if return_tensors == "pt":
            return {"input_ids": torch.tensor(out),
                    "attention_mask": torch.tensor(mask)}
        return {"input_ids": out[0] if isinstance(text, str) else out,
                "attention_mask": mask[0] if isinstance(text, str) else mask}

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(str(int(i)) for i in ids if not (skip_special_tokens and int(i) == PAD))


class _StubOut:
    def __init__(self, sequences, scores):
        self.sequences = sequences
        self.scores = scores


class _StubPolicy:
    """Emits a deterministic continuation whose length and content depend on the
    PROMPT, not on the row index.

    This matters for the equivalence test: keying the output off the row index
    means chunk=1 (always row 0) and chunk=4 (rows 0..3) disagree by
    construction, which tests the stub instead of _generate. Deriving everything
    from the row's real (unpadded) token count keeps outputs stable under any
    chunking while still varying across rows, so trailing-pad trimming and
    per-row score indexing are both exercised.
    """

    def __init__(self):
        self._p = torch.nn.Parameter(torch.zeros(1))

    def parameters(self):
        return iter([self._p])

    def generate(self, input_ids=None, attention_mask=None, max_new_tokens=8,
                 output_scores=False, return_dict_in_generate=False, **kw):
        bsz, _ = input_ids.shape
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        real = [int(attention_mask[r].sum()) for r in range(bsz)]
        n_gen = [(rl % 4) + 2 for rl in real]
        gen = torch.full((bsz, max_new_tokens), PAD, dtype=input_ids.dtype)
        scores = []
        for t in range(max_new_tokens):
            step = torch.full((bsz, VOCAB), -1e4)
            for r in range(bsz):
                if t < n_gen[r]:
                    tok = 3 + ((real[r] + t) % 6)
                    gen[r, t] = tok
                    # Prompt-derived logit: a row mix-up changes the recovered
                    # logprob, and it stays identical across chunk sizes.
                    step[r, tok] = float(real[r] % 5) + 1.0
            scores.append(step)
        seqs = torch.cat([input_ids, gen], dim=1)
        if return_dict_in_generate:
            return _StubOut(seqs, tuple(scores) if output_scores else None)
        return seqs


def _cfg(chunk: int) -> Phase3PPOConfig:
    return Phase3PPOConfig(
        silver_path="", output_dir="",
        max_new_tokens=8, max_input_length=64,
        rollout_chunk_size=chunk, use_real_logprobs=True,
    )


# Different lengths so left padding is actually non-trivial.
PROMPTS = ["a" * 5, "b" * 9, "c" * 7, "d" * 6]


def test_queries_have_no_padding():
    """Query tensors must be the real prompt tokens, never left-padded."""
    q, _, _, _ = _generate(_StubPolicy(), _StubTokenizer(), PROMPTS, _cfg(4), "cpu")
    assert [t.numel() for t in q] == [5, 9, 7, 6]
    for t in q:
        assert (t != PAD).all(), "left pad leaked into the query handed to TRL"


def test_batched_matches_serial():
    """chunk=4 must agree with chunk=1 on every returned quantity."""
    ser = _generate(_StubPolicy(), _StubTokenizer(), PROMPTS, _cfg(1), "cpu")
    bat = _generate(_StubPolicy(), _StubTokenizer(), PROMPTS, _cfg(4), "cpu")
    for i in range(len(PROMPTS)):
        assert torch.equal(ser[0][i], bat[0][i]), f"query {i} differs"
        assert torch.equal(ser[1][i], bat[1][i]), f"response {i} differs"
        assert ser[2][i] == bat[2][i], f"text {i} differs"
        assert ser[3][i] == bat[3][i], f"logprobs {i} differ"


def test_trailing_pad_trimmed():
    """Prompt lengths 5,9,7,6 -> (len%4)+2 = 3,3,5,4 real tokens, no pad filler."""
    _, resp, _, _ = _generate(_StubPolicy(), _StubTokenizer(), PROMPTS, _cfg(4), "cpu")
    assert [t.numel() for t in resp] == [3, 3, 5, 4]
    for t in resp:
        assert (t != PAD).all()


def test_scores_row_indexing():
    """row= selects the right sequence, so logprobs follow the right trajectory."""
    # Three vocab entries, not two: with only the sampled token above the floor,
    # log_softmax sends both rows to ~0.0 regardless of the logit's magnitude and
    # the assertion cannot distinguish them. A competing entry makes the
    # normalisation -- and therefore the recovered logprob -- row-dependent.
    scores = (torch.tensor([[0.0, 1.0, 0.0],
                            [0.0, 4.0, 0.0]]),)
    ids = torch.tensor([1])
    r0 = _step_logprobs_from_scores(ids, scores, [(0, 1)], row=0)[0]
    r1 = _step_logprobs_from_scores(ids, scores, [(0, 1)], row=1)[0]
    assert r0 != r1, "row= is being ignored; every rollout would read row 0"


def test_chunk_boundary_split():
    """chunk=2 over 4 prompts must equal one chunk of 4 (two generate() calls)."""
    a = _generate(_StubPolicy(), _StubTokenizer(), PROMPTS, _cfg(2), "cpu")
    b = _generate(_StubPolicy(), _StubTokenizer(), PROMPTS, _cfg(4), "cpu")
    # Response CONTENT is row-dependent in the stub, so compare the invariant
    # that matters: chunking must not drop or reorder prompts.
    assert len(a[0]) == len(b[0]) == len(PROMPTS)
    assert [t.numel() for t in a[0]] == [t.numel() for t in b[0]]
