# Copyright 2026 Yi Zhang
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
import torch

from verl import DataProto
from verl.trainer.ppo.relay_metrics import compute_relay_metrics, relay_dump_fields

GARBAGE = 100.0  # teacher_logprobs columns that must never be read


def _structure_batch():
    # response lengths 8 / 5 / 6. Mask bits beyond each length are padding and must be ignored.
    response_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 0, 0],
        ]
    )
    relay_teacher_mask = torch.tensor(
        [
            [0, 0, 1, 1, 0, 0, 1, 0],  # segments [2,4) and [6,7)
            [0, 0, 0, 0, 0, 0, 1, 0],  # only a padding bit
            [1, 1, 1, 0, 0, 0, 0, 1],  # segment [0,3) plus a padding bit
        ],
        dtype=torch.bool,
    )
    return DataProto.from_dict(
        tensors={
            "prompts": torch.ones(3, 2, dtype=torch.long),
            "responses": torch.ones(3, 8, dtype=torch.long),
            "response_mask": response_mask,
            "relay_teacher_mask": relay_teacher_mask,
        }
    )


def test_takeover_structure_ignores_padding_bits():
    metrics = compute_relay_metrics(_structure_batch())

    assert metrics["relay/teacher_token_ratio"] == pytest.approx(6 / 19)
    assert metrics["relay/resp_with_teacher_ratio"] == pytest.approx(2 / 3)
    assert metrics["relay/teacher_segments_per_resp_mean"] == pytest.approx(1.0)
    assert metrics["relay/teacher_segments_eq0_ratio"] == pytest.approx(1 / 3)
    assert metrics["relay/teacher_segments_eq1_ratio"] == pytest.approx(1 / 3)
    assert metrics["relay/teacher_segments_eq2_ratio"] == pytest.approx(1 / 3)
    assert metrics["relay/teacher_segments_ge3_ratio"] == pytest.approx(0.0)
    assert metrics["relay/teacher_segment_len_mean"] == pytest.approx(2.0)
    assert metrics["relay/first_teacher_pos_mean"] == pytest.approx(1.0)
    assert metrics["relay/first_teacher_pos_frac_mean"] == pytest.approx(0.125)


def _logprob_batch():
    # P=3, R=4. Response token j's teacher log-prob lives in column P-1+j = 2+j.
    teacher_logprobs = torch.full((2, 7, 1), GARBAGE)
    teacher_logprobs[0, 2:6, 0] = torch.tensor([-1.5, -0.5, -1.0, -0.5])
    teacher_logprobs[1, 2:4, 0] = torch.tensor([-1.0, -3.0])
    return DataProto.from_dict(
        tensors={
            "prompts": torch.ones(2, 3, dtype=torch.long),
            "responses": torch.ones(2, 4, dtype=torch.long),
            "response_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]]),
            "relay_teacher_mask": torch.tensor([[0, 1, 1, 0], [0, 0, 0, 0]], dtype=torch.bool),
            "old_log_probs": torch.tensor([[-1.0, -4.0, -3.0, -0.5], [-2.0, -1.0, 0.0, 0.0]]),
            "teacher_logprobs": teacher_logprobs,
            "rollout_log_probs": torch.tensor([[-1.5, -0.5, -1.0, -0.4], [-1.0, -3.2, 0.0, 0.0]]),
        }
    )


def test_k1_is_split_by_leg_with_teacher_logprobs_shifted_by_one():
    # k1 = old - teacher: s0 [0.5, -3.5, -2.0, 0.0], s1 [-1.0, 2.0]; teacher leg = s0 positions 1-2
    metrics = compute_relay_metrics(_logprob_batch())

    assert metrics["opd/k1/all_tokens_mean"] == pytest.approx(-4.0 / 6)
    assert metrics["opd/k1/student_leg_mean"] == pytest.approx(0.375)
    assert metrics["opd/k1/teacher_leg_mean"] == pytest.approx(-2.75)
    assert metrics["opd/k1/trigger_token_mean"] == pytest.approx(-3.5)


def test_entropy_is_split_by_leg():
    entropys = torch.tensor([[1.0, 2.0, 4.0, 3.0], [0.5, 1.5, 9.0, 9.0]])

    metrics = compute_relay_metrics(_logprob_batch(), entropys=entropys)

    assert metrics["opd/entropy/student_leg_mean"] == pytest.approx(1.5)
    assert metrics["opd/entropy/teacher_leg_mean"] == pytest.approx(3.0)


def test_rollout_logprob_source_diagnostics():
    metrics = compute_relay_metrics(_logprob_batch())

    assert metrics["opd/debug/rollout_vs_teacher_lp_absdiff"] == pytest.approx(0.05)
    assert metrics["opd/debug/rollout_vs_student_lp_absdiff"] == pytest.approx(1.55)


def test_score_is_split_by_budget_exhaustion():
    batch = DataProto.from_dict(
        tensors={
            "prompts": torch.ones(4, 1, dtype=torch.long),
            "responses": torch.ones(4, 2, dtype=torch.long),
            "response_mask": torch.ones(4, 2),
            "relay_teacher_mask": torch.zeros(4, 2, dtype=torch.bool),
            "token_level_scores": torch.tensor([[0.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]]),
        },
        non_tensors={"trigger_stopped": np.array([True, False, None, "true"], dtype=object)},
    )

    metrics = compute_relay_metrics(batch)

    assert metrics["relay/score_mean_budget_exhausted"] == pytest.approx(0.0)
    assert metrics["relay/score_mean_not_exhausted"] == pytest.approx(0.5)


def test_plain_opd_batch_reports_only_student_leg():
    teacher_logprobs = torch.full((1, 4, 1), GARBAGE)
    teacher_logprobs[0, 0:2, 0] = torch.tensor([-2.0, -2.5])
    batch = DataProto.from_dict(
        tensors={
            "prompts": torch.ones(1, 1, dtype=torch.long),
            "responses": torch.ones(1, 3, dtype=torch.long),
            "response_mask": torch.tensor([[1, 1, 0]]),
            "old_log_probs": torch.tensor([[-1.0, -2.0, 0.0]]),
            "teacher_logprobs": teacher_logprobs,
        }
    )

    metrics = compute_relay_metrics(batch)

    assert metrics["opd/k1/student_leg_mean"] == pytest.approx(0.75)
    assert not any(key.startswith("relay/") for key in metrics)
    assert "opd/k1/teacher_leg_mean" not in metrics


def test_k1_is_skipped_for_right_padded_offline_prompts():
    # Offline-trajectory batches right-pad prompts, so column P-1 is not the predictor of response token 0.
    batch = DataProto.from_dict(
        tensors={
            "prompts": torch.tensor([[7, 0]]),
            "responses": torch.ones(1, 2, dtype=torch.long),
            "attention_mask": torch.tensor([[1, 0, 1, 1]]),
            "response_mask": torch.ones(1, 2),
            "old_log_probs": torch.tensor([[-1.0, -2.0]]),
            "teacher_logprobs": torch.zeros(1, 4, 1),
        }
    )

    metrics = compute_relay_metrics(batch)

    assert not any(key.startswith("opd/k1/") for key in metrics)


def test_mask_with_wrong_width_is_rejected():
    batch = DataProto.from_dict(
        tensors={
            "prompts": torch.ones(1, 1, dtype=torch.long),
            "responses": torch.ones(1, 3, dtype=torch.long),
            "response_mask": torch.ones(1, 3),
            "relay_teacher_mask": torch.zeros(1, 4, dtype=torch.bool),
        }
    )

    with pytest.raises(ValueError, match="relay_teacher_mask"):
        compute_relay_metrics(batch)


class _CharTokenizer:
    """Test double: one character per token id; id 0 is a special pad token."""

    vocab = {0: "<pad>", 1: "a", 2: "b", 3: "c", 4: "d", 5: "e"}

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.vocab[int(i)] for i in ids if not (skip_special_tokens and int(i) == 0))


def test_dump_fields_mark_teacher_spans():
    batch = DataProto.from_dict(
        tensors={
            "prompts": torch.ones(2, 1, dtype=torch.long),
            "responses": torch.tensor([[1, 2, 3, 4, 5], [1, 2, 3, 0, 0]]),
            "response_mask": torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]]),
            "relay_teacher_mask": torch.tensor([[0, 1, 1, 0, 1], [0, 0, 0, 1, 0]], dtype=torch.bool),
        },
        non_tensors={
            "trigger_stopped": np.array([True, None], dtype=object),
            "trigger_stop_pos": np.array([5, None], dtype=object),
        },
    )

    fields = relay_dump_fields(batch, _CharTokenizer())

    assert fields["relay_teacher_spans"] == [[[1, 3], [4, 5]], []]
    assert fields["relay_teacher_tokens"] == [3, 0]
    assert fields["relay_output_marked"] == ["a【T】bc【/T】d【T】e【/T】", "abc"]
    assert fields["trigger_stopped"] == [True, False]
    assert fields["trigger_stop_pos"] == [5, None]


def test_dump_fields_are_empty_without_relay_mask():
    batch = DataProto.from_dict(
        tensors={
            "prompts": torch.ones(1, 1, dtype=torch.long),
            "responses": torch.tensor([[1, 2]]),
            "response_mask": torch.ones(1, 2),
        }
    )

    assert relay_dump_fields(batch, _CharTokenizer()) == {}
