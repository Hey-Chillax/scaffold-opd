# Modified by Yi Zhang, 2026.
from types import SimpleNamespace

import pytest
import torch

from opd.patches.vllm import speculative_decode


def test_trigger_stop_profile_without_draft_probs_uses_original_sampler(monkeypatch):
    monkeypatch.setenv("VERL_OPD_ROLLOUT_MODE", "trigger_stop")
    monkeypatch.setenv("RELAY_OPD_TRIGGER_TOPK", "5")
    monkeypatch.setattr(
        speculative_decode,
        "_get_reflection_token_ids",
        lambda device: torch.tensor([1], dtype=torch.long, device=device),
    )

    expected = torch.tensor([[7, -1]], dtype=torch.int32)
    calls = []

    def fake_original(*args, **kwargs):
        calls.append((args, kwargs))
        return expected

    monkeypatch.setattr(speculative_decode, "_standard_rejection_sample_output", fake_original)

    output = speculative_decode._relay_opd_rejection_sample(
        draft_token_ids=torch.tensor([0], dtype=torch.int32),
        num_draft_tokens=[1],
        max_spec_len=1,
        cu_num_draft_tokens=torch.tensor([0, 1], dtype=torch.int32),
        draft_probs=None,
        target_logits=torch.zeros((1, 8), dtype=torch.float32),
        bonus_token_ids=torch.tensor([0], dtype=torch.int32),
        sampling_metadata=SimpleNamespace(),
    )

    assert output is expected
    assert len(calls) == 1


def test_trigger_stop_real_request_without_draft_probs_fails(monkeypatch):
    monkeypatch.setenv("VERL_OPD_ROLLOUT_MODE", "trigger_stop")
    monkeypatch.setenv("RELAY_OPD_TRIGGER_TOPK", "5")
    monkeypatch.setattr(
        speculative_decode,
        "_get_reflection_token_ids",
        lambda device: torch.tensor([1], dtype=torch.long, device=device),
    )

    with pytest.raises(RuntimeError, match="Relay-OPD trigger detection requires draft_probs"):
        speculative_decode._relay_opd_rejection_sample(
            draft_token_ids=torch.tensor([0], dtype=torch.int32),
            num_draft_tokens=[1],
            max_spec_len=1,
            cu_num_draft_tokens=torch.tensor([0, 1], dtype=torch.int32),
            draft_probs=None,
            target_logits=torch.zeros((1, 8), dtype=torch.float32),
            bonus_token_ids=torch.tensor([0], dtype=torch.int32),
            sampling_metadata=SimpleNamespace(req_ids=["request-00000000"]),
        )


class _FakeTriggerTokenizer:
    # Qwen3 ids for the strings the header-conclude rule resolves; " ✅" is two byte-level pieces.
    _VOCAB = {" Final": [13023], "###": [14374], " ###": [16600], " ✅": [25521, 227]}

    def encode(self, text, add_special_tokens=False):
        return list(self._VOCAB.get(text, [1, 2]))


@pytest.fixture
def fake_trigger_tokenizer(monkeypatch):
    monkeypatch.setattr(speculative_decode, "_load_trigger_tokenizer", lambda: _FakeTriggerTokenizer())
    speculative_decode._HEADER_CONCLUDE_CACHE.clear()
    yield
    speculative_decode._HEADER_CONCLUDE_CACHE.clear()


def test_header_conclude_rule_follows_the_trigger_list(fake_trigger_tokenizer, monkeypatch):
    monkeypatch.delenv("RELAY_OPD_HEADER_CONCLUDE", raising=False)
    cpu = torch.device("cpu")

    ids = speculative_decode._get_header_conclude_ids(cpu, torch.tensor([14190, 13023]))
    assert ids is not None
    header, decoration = ids
    assert header.tolist() == [14374, 16600]
    assert decoration.tolist() == [25521]

    # The stock reflection list has no " Final", so the rule stays off for plain Relay-OPD.
    assert speculative_decode._get_header_conclude_ids(cpu, torch.tensor([14190, 13824])) is None


def test_header_conclude_rule_env_override(fake_trigger_tokenizer, monkeypatch):
    cpu = torch.device("cpu")
    monkeypatch.setenv("RELAY_OPD_HEADER_CONCLUDE", "1")
    assert speculative_decode._get_header_conclude_ids(cpu, torch.tensor([14190])) is not None
    monkeypatch.setenv("RELAY_OPD_HEADER_CONCLUDE", "0")
    assert speculative_decode._get_header_conclude_ids(cpu, torch.tensor([13023])) is None


def test_previous_token_ids_follow_the_flattened_layout():
    # Two requests with 3 and 2 drafts, flattened as [a0 a1 a2 b0 b1]. A request's first draft position takes
    # its last token from the previous step; request b has none yet.
    cpu = torch.device("cpu")
    draft = torch.tensor([5, 6, 7, 8, 9])
    _, _, req_ids, pos_in_req, valid_pos = speculative_decode._request_layout([3, 2], 5, 4, torch.tensor([3, 5]), cpu)
    prev = speculative_decode._previous_token_ids(draft, req_ids, pos_in_req, valid_pos, torch.tensor([100, -1]))
    assert prev.tolist() == [100, 5, 6, -1, 8]


def test_remember_last_emitted_skips_placeholders_and_drops_stopped_requests(monkeypatch):
    monkeypatch.setattr(speculative_decode, "_LAST_EMITTED_TOKEN", {"c": 7})
    placeholder = speculative_decode.PLACEHOLDER_TOKEN_ID
    output = torch.tensor([[11, 12, placeholder], [21, placeholder, placeholder], [31, 32, 99]], dtype=torch.int32)
    internal_stop = torch.zeros_like(output, dtype=torch.bool)
    internal_stop[2, 2] = True

    speculative_decode._remember_last_emitted(output, internal_stop, ["a", "b", "c"])

    assert speculative_decode._LAST_EMITTED_TOKEN == {"a": 12, "b": 21}
