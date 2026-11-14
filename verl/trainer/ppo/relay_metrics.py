# Copyright 2026 Yi Zhang
# SPDX-License-Identifier: Apache-2.0
"""Driver-side diagnostics for Relay-OPD and plain OPD training batches.

The statistics are exact values over the whole gathered batch (no per-micro-batch
averaging), computed on CPU after the actor update.

Tensor layouts (B = batch size, P = padded prompt width, R = padded response width):
- ``prompts`` [B, P] is left-padded. ``responses``, ``response_mask``, ``old_log_probs``,
  ``rollout_log_probs`` and ``entropys`` are [B, R], right-padded and indexed by
  response-relative position.
- ``relay_teacher_mask`` [B, R] bool is True where the token was emitted on a teacher leg.
  A teacher leg starts with its trigger token, so every run of True opens with a trigger.
  Two takeovers that touch merge into one run, so runs ("segments") lower-bound takeovers.
- ``teacher_logprobs`` [B, P+R, K] (K=1 for estimator losses) and ``teacher_token_logprobs``
  [B, P+R] or [B, P+R, 1] hold teacher prompt log-probs: entry k scores token k+1 of the
  left-padded prompt|response sequence, so response token j sits in column P-1+j.
"""

from typing import Any, Optional

import numpy as np
import torch

from verl import DataProto

TEACHER_SPAN_OPEN = "【T】"
TEACHER_SPAN_CLOSE = "【/T】"


def _truthy(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _maybe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        if np.isnan(value):
            return None
    except TypeError:
        pass
    return int(value)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    # Boolean indexing keeps padded entries (which may be garbage or -inf) out of the sum.
    return float(values.to(torch.float64)[mask].mean())


def _segment_starts(teacher_mask: torch.Tensor) -> torch.Tensor:
    """[B, R] bool, True at the first token of every run of teacher tokens."""
    starts = teacher_mask.clone()
    starts[:, 1:] = teacher_mask[:, 1:] & ~teacher_mask[:, :-1]
    return starts


def _check_response_shape(name: str, tensor: torch.Tensor, response_mask: torch.Tensor) -> None:
    if tuple(tensor.shape) != tuple(response_mask.shape):
        raise ValueError(
            f"{name} shape {tuple(tensor.shape)} does not match response_mask shape {tuple(response_mask.shape)}"
        )


def _teacher_mask(batch: DataProto, response_mask: torch.Tensor) -> Optional[torch.Tensor]:
    if "relay_teacher_mask" not in batch.batch.keys():
        return None
    raw = batch.batch["relay_teacher_mask"]
    _check_response_shape("relay_teacher_mask", raw, response_mask)
    return raw.detach().cpu().bool() & response_mask


def _response_teacher_logprobs(batch: DataProto, response_mask: torch.Tensor) -> Optional[torch.Tensor]:
    """Teacher log-prob of each emitted response token, [B, R], or None if unavailable."""
    if "teacher_token_logprobs" in batch.batch.keys():
        scores = batch.batch["teacher_token_logprobs"]
    elif "teacher_logprobs" in batch.batch.keys() and batch.batch["teacher_logprobs"].shape[-1] == 1:
        # Estimator losses request K=1, i.e. the log-prob of the actual next token.
        scores = batch.batch["teacher_logprobs"]
    else:
        # Top-k scores without the actual-token log-prob cannot give per-token k1.
        return None
    if scores.dim() == 3:
        if scores.shape[-1] != 1:
            raise ValueError(f"expected one teacher log-prob per position, got shape {tuple(scores.shape)}")
        scores = scores[..., 0]
    prompt_width = batch.batch["prompts"].shape[1]
    response_width = response_mask.shape[1]
    if prompt_width < 1:
        raise ValueError("teacher log-prob alignment needs a non-empty prompt")
    attention_mask = batch.batch.get("attention_mask", None)
    if attention_mask is not None and not bool(attention_mask[:, prompt_width - 1].bool().all()):
        # Offline-trajectory batches right-pad prompts, so column P-1+j does not score response token j.
        return None
    if scores.shape[1] != prompt_width + response_width:
        raise ValueError(
            f"teacher log-prob width {scores.shape[1]} != prompt width {prompt_width} + response width {response_width}"
        )
    return scores[:, prompt_width - 1 : prompt_width - 1 + response_width].detach().cpu().to(torch.float64)


def _takeover_structure_metrics(teacher_mask: torch.Tensor, response_mask: torch.Tensor) -> dict[str, float]:
    resp_lens = response_mask.sum(dim=-1)
    rows = resp_lens > 0
    if not rows.any():
        return {}
    starts = _segment_starts(teacher_mask)
    segments = starts.sum(dim=-1)[rows].to(torch.float64)
    has_teacher = teacher_mask.any(dim=-1)
    n_teacher = int(teacher_mask.sum())
    metrics = {
        "relay/teacher_token_ratio": n_teacher / int(response_mask.sum()),
        "relay/resp_with_teacher_ratio": float(has_teacher[rows].to(torch.float64).mean()),
        "relay/teacher_segments_per_resp_mean": float(segments.mean()),
        "relay/teacher_segments_eq0_ratio": float((segments == 0).to(torch.float64).mean()),
        "relay/teacher_segments_eq1_ratio": float((segments == 1).to(torch.float64).mean()),
        "relay/teacher_segments_eq2_ratio": float((segments == 2).to(torch.float64).mean()),
        "relay/teacher_segments_ge3_ratio": float((segments >= 3).to(torch.float64).mean()),
    }
    n_segments = int(starts.sum())
    if n_segments > 0:
        # argmax returns the first maximal index, i.e. the first teacher token of each row.
        first_pos = teacher_mask.to(torch.int32).argmax(dim=-1)[has_teacher].to(torch.float64)
        metrics["relay/teacher_segment_len_mean"] = n_teacher / n_segments
        metrics["relay/first_teacher_pos_mean"] = float(first_pos.mean())
        metrics["relay/first_teacher_pos_frac_mean"] = float(
            (first_pos / resp_lens[has_teacher].to(torch.float64)).mean()
        )
    return metrics


def _budget_split_metrics(batch: DataProto) -> dict[str, float]:
    """Training score split by whether the response ended after the M-th takeover."""
    flags_raw = batch.non_tensor_batch.get("trigger_stopped", None)
    if flags_raw is None or "token_level_scores" not in batch.batch.keys():
        return {}
    exhausted = torch.tensor([_truthy(v) for v in flags_raw], dtype=torch.bool)
    scores = batch.batch["token_level_scores"].detach().cpu().to(torch.float64).sum(dim=-1)
    metrics = {}
    if exhausted.any():
        metrics["relay/score_mean_budget_exhausted"] = float(scores[exhausted].mean())
    if (~exhausted).any():
        metrics["relay/score_mean_not_exhausted"] = float(scores[~exhausted].mean())
    return metrics


def compute_relay_metrics(batch: DataProto, entropys: Optional[torch.Tensor] = None) -> dict[str, float]:
    """Relay takeover structure (``relay/*``) and per-leg OPD signals (``opd/*``).

    ``opd/k1/*`` uses k1 = log pi_old(y_t) - log pi_T(y_t) on the emitted token y_t. On student
    legs its mean estimates the per-token reverse KL; on teacher legs (y_t ~ pi_T) it estimates
    minus the forward KL. Batches without ``relay_teacher_mask`` (plain OPD) count every response
    token as student leg. ``entropys`` is the [B, R] student entropy from the old-log-prob pass.
    """
    response_mask = batch.batch["response_mask"].detach().cpu().bool()
    teacher_mask = _teacher_mask(batch, response_mask)
    student_mask = response_mask if teacher_mask is None else response_mask & ~teacher_mask
    has_teacher_tokens = teacher_mask is not None and bool(teacher_mask.any())

    metrics: dict[str, float] = {}
    if teacher_mask is not None:
        metrics.update(_takeover_structure_metrics(teacher_mask, response_mask))
        metrics.update(_budget_split_metrics(batch))

    teacher_lp = _response_teacher_logprobs(batch, response_mask)
    old_lp = batch.batch.get("old_log_probs", None)
    if old_lp is not None:
        _check_response_shape("old_log_probs", old_lp, response_mask)
        old_lp = old_lp.detach().cpu().to(torch.float64)

    if teacher_lp is not None and old_lp is not None:
        k1 = old_lp - teacher_lp
        if response_mask.any():
            metrics["opd/k1/all_tokens_mean"] = _masked_mean(k1, response_mask)
        if student_mask.any():
            metrics["opd/k1/student_leg_mean"] = _masked_mean(k1, student_mask)
        if has_teacher_tokens:
            metrics["opd/k1/teacher_leg_mean"] = _masked_mean(k1, teacher_mask)
            metrics["opd/k1/trigger_token_mean"] = _masked_mean(k1, _segment_starts(teacher_mask))

    if entropys is not None:
        _check_response_shape("entropys", entropys, response_mask)
        entropys = entropys.detach().cpu().to(torch.float64)
        if student_mask.any():
            metrics["opd/entropy/student_leg_mean"] = _masked_mean(entropys, student_mask)
        if has_teacher_tokens:
            metrics["opd/entropy/teacher_leg_mean"] = _masked_mean(entropys, teacher_mask)

    # Which policy scored rollout_log_probs? Under relay speculative decoding the vLLM target is the
    # teacher, so these should track the teacher pool rather than the student.
    rollout_lp = batch.batch.get("rollout_log_probs", None)
    if rollout_lp is not None and response_mask.any():
        _check_response_shape("rollout_log_probs", rollout_lp, response_mask)
        rollout_lp = rollout_lp.detach().cpu().to(torch.float64)
        if teacher_lp is not None:
            metrics["opd/debug/rollout_vs_teacher_lp_absdiff"] = _masked_mean(
                (rollout_lp - teacher_lp).abs(), response_mask
            )
        if old_lp is not None:
            metrics["opd/debug/rollout_vs_student_lp_absdiff"] = _masked_mean(
                (rollout_lp - old_lp).abs(), response_mask
            )
    return metrics


def relay_dump_fields(batch: DataProto, tokenizer) -> dict[str, list]:
    """Per-sample relay fields for the rollout JSONL dump (``trainer.rollout_data_dir``).

    ``relay_teacher_spans`` lists response-relative [start, end) token ranges of teacher legs, and
    ``relay_output_marked`` is the decoded response with each teacher leg wrapped in
    TEACHER_SPAN_OPEN / TEACHER_SPAN_CLOSE. Returns {} for batches without a relay mask.
    """
    response_mask = batch.batch["response_mask"].detach().cpu().bool()
    teacher_mask = _teacher_mask(batch, response_mask)
    if teacher_mask is None:
        return {}
    responses = batch.batch["responses"].detach().cpu()

    spans, counts, marked = [], [], []
    for row in range(responses.shape[0]):
        valid = response_mask[row].nonzero().flatten()
        length = int(valid.max()) + 1 if valid.numel() else 0
        ids = responses[row, :length].tolist()
        is_teacher = teacher_mask[row, :length].tolist()
        row_spans, pieces = [], []
        start = 0
        while start < length:
            end = start
            while end < length and is_teacher[end] == is_teacher[start]:
                end += 1
            text = tokenizer.decode(ids[start:end], skip_special_tokens=True)
            if is_teacher[start]:
                row_spans.append([start, end])
                pieces.append(f"{TEACHER_SPAN_OPEN}{text}{TEACHER_SPAN_CLOSE}")
            else:
                pieces.append(text)
            start = end
        spans.append(row_spans)
        counts.append(sum(end - start for start, end in row_spans))
        marked.append("".join(pieces))

    fields: dict[str, list] = {
        "relay_teacher_spans": spans,
        "relay_teacher_tokens": counts,
        "relay_output_marked": marked,
    }
    flags = batch.non_tensor_batch.get("trigger_stopped", None)
    if flags is not None:
        fields["trigger_stopped"] = [_truthy(v) for v in flags]
    positions = batch.non_tensor_batch.get("trigger_stop_pos", None)
    if positions is not None:
        fields["trigger_stop_pos"] = [_maybe_int(v) for v in positions]
    return fields
