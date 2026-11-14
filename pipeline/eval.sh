#!/usr/bin/env bash
# Copyright 2026 Yi Zhang
# SPDX-License-Identifier: Apache-2.0
#
# Student-only evaluation of one model with opd/eval/math_benchmarks.py on one GPU (the first visible one).
#
#   MODEL=<HuggingFace model dir> OUT_DIR=<dir> bash pipeline/eval.sh
#
# The defaults are the settings behind every reported number: temperature 1.0, top-p 1.0, up to 32,768 new
# tokens, the Qwen3 non-thinking chat template, 4 samples per MATH-500 problem and 16 per AIME problem.
# It calls the evaluator directly rather than opd/scripts/evaluation/math.sh, which pins its shards to GPUs
# 0..N-1 and would ignore the GPU a scheduler assigned.
#
# Env (see pipeline/common.sh for DATA_DIR, OUTPUT_ROOT, VENV, ENV_SETUP, DRY_RUN):
#   MODEL            required, HuggingFace model directory
#   OUT_DIR          default $OUTPUT_ROOT/eval/<EVAL_TAG>, EVAL_TAG defaulting to a timestamp
#   BENCHES          comma-separated, default math500,aime24,aime25,aime26
#   BENCH_N_SAMPLES  samples per problem, parallel to BENCHES; default 4,16,16,16 for the default BENCHES,
#                    otherwise N_SAMPLES (default 1) for every benchmark
#   MAX_NEW          max new tokens, default 32768
#   N_PROBLEMS       limit unique problems per benchmark, default -1 (all)
#   BENCH_DIR        default $DATA_DIR/bench (pipeline/prepare_data.py)
#   EVAL_EXIT_GRACE  seconds the evaluator gets to exit by itself after its last summary is written, default 300
#   EVAL_EXTRA_ARGS  further math_benchmarks.py arguments, split on spaces
set -eo pipefail

# shellcheck source=pipeline/common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
LOG_TAG="eval"
setup_env
set -u
cd "$REPO_ROOT"
export MATH_GRADER_PATH=${MATH_GRADER_PATH:-$REPO_ROOT/opd/reward/grader}

MODEL=${MODEL:?set MODEL to a HuggingFace model directory}
require_path "$MODEL/config.json"
benches=${BENCHES:-math500,aime24,aime25,aime26}
if [[ -n "${BENCH_N_SAMPLES:-}" ]]; then
    samples=(--bench_n_samples "$BENCH_N_SAMPLES")
elif [[ -z "${BENCHES:-}" ]]; then
    samples=(--bench_n_samples "4,16,16,16")
else
    samples=(--n_samples "${N_SAMPLES:-1}")
fi
max_new=${MAX_NEW:-32768}
[[ "$max_new" =~ ^[1-9][0-9]*$ ]] || die "MAX_NEW must be a positive integer, got '$max_new'"
out_dir=${OUT_DIR:-$OUTPUT_ROOT/eval/${EVAL_TAG:-$(date +%Y%m%d_%H%M%S)}}
bench_dir=${BENCH_DIR:-$DEFAULT_BENCH_DIR}
IFS=, read -r -a bench_list <<<"$benches"
for bench in "${bench_list[@]}"; do
    case "$bench" in
        math500) require_path "$bench_dir/math500.parquet" ;;
        aime24) require_path "$bench_dir/aime-24.parquet" ;;
        aime25) require_path "$bench_dir/aime-2025.parquet" ;;
        aime26) require_path "$bench_dir/aime26.parquet" ;;
    esac
done

# shellcheck disable=SC2206  # EVAL_EXTRA_ARGS is a space-separated list on purpose
extra=(${EVAL_EXTRA_ARGS:-})
cmd=(python3 opd/eval/math_benchmarks.py
    --model "$MODEL"
    --benches "$benches"
    --data_dir "$bench_dir"
    "${samples[@]}"
    --n_problems "${N_PROBLEMS:--1}"
    --max_new "$max_new"
    --max_model_len $((2048 + max_new + 1))
    --temperature 1.0
    --top_p 1.0
    --tp 1
    --gpu_mem 0.85
    --seed 42
    --disable_thinking
    --out_dir "$out_dir"
    ${extra[@]+"${extra[@]}"})

log "model=$MODEL benches=$benches samples=${samples[*]} max_new=$max_new out=$out_dir gpu=${CUDA_VISIBLE_DEVICES:-unset}"
if is_dry_run; then
    run "${cmd[@]}"
    exit 0
fi

mkdir -p "$out_dir"
# Only summaries newer than this run count, so a reused OUT_DIR cannot end a run early. The marker is per process
# because two evaluations can share an OUT_DIR, each with its own benchmarks.
started=$out_dir/.eval_started.$$
touch "$started"
trap 'rm -f "$started"' EXIT
all_summaries_written() {
    local b
    for b in "${bench_list[@]}"; do
        [[ "$out_dir/$b.summary.json" -nt "$started" ]] || return 1
    done
}

# The evaluator can hang at interpreter exit after writing every summary. setsid gives it its own process group,
# which vLLM's engine processes join, so the group can be ended once the results are on disk and the GPU can go
# to the next phase.
setsid "${cmd[@]}" &
eval_pid=$!

grace=${EVAL_EXIT_GRACE:-300}
written_at=-1
while kill -0 "$eval_pid" 2>/dev/null; do
    if all_summaries_written; then
        ((written_at >= 0)) || written_at=$SECONDS
        if ((SECONDS - written_at >= grace)); then
            log "every summary is written but the evaluator is still up after ${grace}s; ending its process group"
            kill -TERM -- "-$eval_pid" 2>/dev/null || true
            sleep 30
            kill -KILL -- "-$eval_pid" 2>/dev/null || true
            break
        fi
    fi
    sleep 15
done

status=0
wait "$eval_pid" || status=$?
if ! all_summaries_written; then
    die "the evaluator exited with status $status before writing every summary"
fi
log "done. Per-benchmark summaries: $out_dir/<bench>.summary.json"
