#!/usr/bin/env bash
# Copyright 2026 Yi Zhang
# SPDX-License-Identifier: Apache-2.0
#
# Inference-only probe of concluding takeovers, before any training. A trained student generates with the teacher
# as its speculative-decoding target, and the teacher takes over only when it wants to start the final-answer header
# while none of the trigger tokens is in the student's top-K; it then writes the rest of the solution. Compare with
# the same student alone at the same budget using tools/analysis/conclude_probe_stats.py.
#
#   STUDENT=<run>/global_step_40/actor/huggingface bash pipeline/conclude_probe.sh
#
# Env (see pipeline/common.sh for paths):
#   STUDENT          required, HuggingFace directory of the trained student
#   TEACHER          default $MODEL_DIR/Qwen3-4B-Instruct-2507
#   TRIGGER_TOKENS   comma-separated, leading spaces count, default "Final, Final,Answer, Answer". With " Final"
#                    in the list the "### ✅" header rule is on as well; RELAY_OPD_HEADER_CONCLUDE=0 turns it off.
#   MAX_NEW          budget, default 16384 (at 1.7B, unfinished responses still exist there)
#   N_AIME / N_MATH  samples per problem, default 8 / 2
#   OUT_DIR          default $OUTPUT_ROOT/conclude_probe_<timestamp>
set -uo pipefail

# shellcheck source=pipeline/common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
LOG_TAG=probe
setup_env
cd "$REPO_ROOT" || exit 1

STUDENT=${STUDENT:?set STUDENT to the trained student}
TEACHER=${TEACHER:-$DEFAULT_TEACHER}
export MAX_NEW=${MAX_NEW:-16384}
n_aime=${N_AIME:-8}
n_math=${N_MATH:-2}
root=${OUT_DIR:-$OUTPUT_ROOT/conclude_probe_$(date +%Y%m%d_%H%M%S)}
require_path "$STUDENT/config.json"
require_path "$TEACHER/config.json"
is_dry_run || mkdir -p "$root"

# The evaluator's worker processes apply the speculative-decoding patch at interpreter start.
export PYTHONPATH=$REPO_ROOT/opd/patches/vllm:${PYTHONPATH:-}
export VERL_OPD_VLLM_PATCH=1
export RELAY_OPD_REFLECTION_TOKENS=${TRIGGER_TOKENS:-Final, Final,Answer, Answer}
export RELAY_OPD_MAX_TAKEOVER_TOKENS=1024
export EVAL_EXTRA_ARGS="--draft_model $STUDENT --rollout_mode relay --max_takeovers 1 --paragraphs_per_takeover 10 --trigger_topk 5 --save_rollout_trace --trace_mode mask"

IFS=, read -r -a gpus <<<"${CUDA_VISIBLE_DEVICES:-0}"
gpu_b=${gpus[1]:-${gpus[0]}}
log "student=$STUDENT teacher=$TEACHER max_new=$MAX_NEW trigger='$RELAY_OPD_REFLECTION_TOKENS' out=$root"
status=0
if is_dry_run || ((${#gpus[@]} < 2)); then
    env CUDA_VISIBLE_DEVICES="${gpus[0]}" MODEL="$TEACHER" OUT_DIR="$root" BENCHES=aime24,aime25 \
        BENCH_N_SAMPLES="$n_aime,$n_aime" bash pipeline/eval.sh || status=1
    env CUDA_VISIBLE_DEVICES="$gpu_b" MODEL="$TEACHER" OUT_DIR="$root" BENCHES=aime26,math500 \
        BENCH_N_SAMPLES="$n_aime,$n_math" bash pipeline/eval.sh || status=1
else
    env CUDA_VISIBLE_DEVICES="${gpus[0]}" MODEL="$TEACHER" OUT_DIR="$root" BENCHES=aime24,aime25 \
        BENCH_N_SAMPLES="$n_aime,$n_aime" bash pipeline/eval.sh >"$root/eval_a.log" 2>&1 &
    pid_a=$!
    env CUDA_VISIBLE_DEVICES="$gpu_b" MODEL="$TEACHER" OUT_DIR="$root" BENCHES=aime26,math500 \
        BENCH_N_SAMPLES="$n_aime,$n_math" bash pipeline/eval.sh >"$root/eval_b.log" 2>&1 &
    pid_b=$!
    wait "$pid_a" || status=1
    wait "$pid_b" || status=1
fi
((status == 0)) || die "evaluation failed; see $root/eval_[ab].log"
log "done; results in $root"
