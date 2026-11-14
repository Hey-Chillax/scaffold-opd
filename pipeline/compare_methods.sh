#!/usr/bin/env bash
# Copyright 2026 Yi Zhang
# SPDX-License-Identifier: Apache-2.0
#
# Train several methods from the same untrained student, then evaluate the student and every saved checkpoint:
#   1. one training run per method in METHODS, one after another (pipeline/train_stage.sh)
#   2. student-only evaluation (pipeline/eval.sh) of the untrained student and of each checkpoint, one model per
#      GPU, as many at a time as there are GPUs
# A failed phase does not stop the later ones; the script exits non-zero if any phase failed.
#
#   bash pipeline/compare_methods.sh                       # Takeover OPD and OPD, 60 steps each
#   METHODS=grpo ROLLOUT_N=8 bash pipeline/compare_methods.sh
#   DRY_RUN=1 bash pipeline/compare_methods.sh
#
# Env (everything train_stage.sh and eval.sh read passes through; see common.sh for paths):
#   METHODS          training runs, in order, default "takeover opd"
#   STEPS            default 60
#   BATCH            default 128
#   SAVE_FREQ        default 20, so each run keeps the checkpoints of steps 20, 40 and 60
#   EVAL_STEPS       checkpoints to evaluate, default "40 60"
#   EVAL_BASE        1 (default) also evaluates the untrained student, with BASE_BENCH_N_SAMPLES
#   BASE_BENCH_N_SAMPLES   default 4,32,32,32 (more AIME samples for the reference row)
#   RUN_ID           names $OUTPUT_ROOT/compare_<RUN_ID>/, default a timestamp
# Outputs: $OUTPUT_ROOT/compare_<RUN_ID>/<method>/ per run and eval/<tag>/ plus eval/<tag>.log per model.
set -uo pipefail

# shellcheck source=pipeline/common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
LOG_TAG=compare
setup_env
cd "$REPO_ROOT" || exit 1

run_id=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
root=$OUTPUT_ROOT/compare_$run_id
methods=${METHODS:-takeover opd}
export STEPS=${STEPS:-60} BATCH=${BATCH:-128} SAVE_FREQ=${SAVE_FREQ:-20}
export STUDENT_MODEL=${STUDENT_MODEL:-$DEFAULT_STUDENT}
read -r -a eval_steps <<<"${EVAL_STEPS:-40 60}"
failed=()
log "run=$run_id methods='$methods' steps=$STEPS batch=$BATCH save_freq=$SAVE_FREQ gpus=${CUDA_VISIBLE_DEVICES:-unset}"

declare -A run_dir ckpt_dir
for method in $methods; do
    run_dir[$method]=$root/$method
    # With CKPT_DIR set (e.g. a fast local disk), each method gets its own subdirectory there.
    ckpt_dir[$method]=${CKPT_DIR:+$CKPT_DIR/$method}
    ckpt_dir[$method]=${ckpt_dir[$method]:-${run_dir[$method]}}
    log "training $method -> ${run_dir[$method]} (checkpoints: ${ckpt_dir[$method]})"
    if env METHOD="$method" OUTPUT_DIR="${run_dir[$method]}" CKPT_DIR="${ckpt_dir[$method]}" \
        bash pipeline/train_stage.sh; then
        log "training $method finished"
    else
        log "ERROR: training $method failed"
        failed+=("train:$method")
    fi
    wait_gpus_idle
done

# Models to evaluate: tags[i] names models[i], with samples[i] per-benchmark sample counts.
tags=()
models=()
samples=()
if [[ "${EVAL_BASE:-1}" == 1 ]]; then
    tags+=(base)
    models+=("$STUDENT_MODEL")
    samples+=("${BASE_BENCH_N_SAMPLES:-4,32,32,32}")
fi
for method in $methods; do
    for s in "${eval_steps[@]}"; do
        model=${ckpt_dir[$method]}/global_step_$s/actor/huggingface
        if is_dry_run || [[ -f "$model/config.json" ]]; then
            tags+=("${method}_step$s")
            models+=("$model")
            samples+=("${BENCH_N_SAMPLES:-4,16,16,16}")
        else
            log "WARNING: no checkpoint at $model; skipping its evaluation"
        fi
    done
done

IFS=, read -r -a gpus <<<"${CUDA_VISIBLE_DEVICES:-0}"
is_dry_run || mkdir -p "$root/eval"
i=0
while ((i < ${#tags[@]})); do
    round_pids=()
    round_tags=()
    for gpu in "${gpus[@]}"; do
        ((i < ${#tags[@]})) || break
        log "evaluating ${tags[i]} on GPU $gpu: ${models[i]}"
        if is_dry_run; then
            env CUDA_VISIBLE_DEVICES="$gpu" MODEL="${models[i]}" OUT_DIR="$root/eval/${tags[i]}" \
                BENCH_N_SAMPLES="${samples[i]}" bash pipeline/eval.sh || failed+=("eval:${tags[i]}")
        else
            env CUDA_VISIBLE_DEVICES="$gpu" MODEL="${models[i]}" OUT_DIR="$root/eval/${tags[i]}" \
                BENCH_N_SAMPLES="${samples[i]}" bash pipeline/eval.sh >"$root/eval/${tags[i]}.log" 2>&1 &
            round_pids+=("$!")
            round_tags+=("${tags[i]}")
        fi
        i=$((i + 1))
    done
    for j in "${!round_pids[@]}"; do
        if wait "${round_pids[j]}"; then
            log "evaluation ${round_tags[j]} finished"
        else
            log "ERROR: evaluation ${round_tags[j]} failed; see $root/eval/${round_tags[j]}.log"
            failed+=("eval:${round_tags[j]}")
        fi
    done
done

if ((${#failed[@]})); then
    log "done with failures: ${failed[*]}"
    exit 1
fi
log "done. Summaries: $root/eval/<tag>/<bench>.summary.json"
