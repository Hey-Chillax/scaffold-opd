#!/usr/bin/env bash
# Copyright 2026 Yi Zhang
# SPDX-License-Identifier: Apache-2.0
#
# Train one stage on one machine: Takeover OPD, plain OPD, GRPO or SFT.
#
#   METHOD=takeover bash pipeline/train_stage.sh [extra Hydra overrides ...]
#   METHOD=opd      bash pipeline/train_stage.sh
#   METHOD=grpo     bash pipeline/train_stage.sh
#   METHOD=sft TRAIN_DATA=<filtered teacher parquet> bash pipeline/train_stage.sh
#   DRY_RUN=1 METHOD=takeover bash pipeline/train_stage.sh      # print the configuration and command only
#
# Takeover OPD and OPD put the student (FSDP actor + vLLM rollout) and the teacher (log-prob prefill) on
# separate GPUs; GRPO and SFT train on every visible GPU. The reported runs used two 80 GB GPUs.
#
# Env (all optional; see pipeline/common.sh for MODEL_DIR, DATA_DIR, OUTPUT_ROOT, VENV, ENV_SETUP, DRY_RUN):
#   METHOD            takeover (default; called "relay" in config keys and metric names) | opd | grpo | sft
#   CONCLUDE_TRIGGER  takeover only: 1 adds the answer-header tokens (" Final", "Answer", and "✅" right after
#                     "###") to the reflection triggers, so the teacher can also take over to conclude
#   STEPS             training steps; default 40 for takeover and opd, 60 for grpo, 0 (one epoch) for sft
#   BATCH             prompts per step, also the PPO mini-batch, default 128
#   ROLLOUT_N         grpo only: rollouts per prompt, default 8
#   RESPONSE_LEN      max response tokens, default 16384
#   STUDENT_MODEL     default $MODEL_DIR/Qwen3-1.7B; a previous stage's huggingface/ directory works too
#   TEACHER_MODEL     default $MODEL_DIR/Qwen3-4B-Instruct-2507
#   TRAIN_DATA        default $DATA_DIR/train/dapo_math_17k_en.parquet (required for sft)
#   BENCH             default $DATA_DIR/bench
#   ACTOR_GPUS_PER_NODE / TEACHER_GPUS_PER_NODE   takeover/opd split, default half of the visible GPUs each
#   OUTPUT_DIR        run directory (training.log, rollout_dump/, wandb/), default
#                     $OUTPUT_ROOT/<method>_s<steps>_b<batch>_<timestamp>; use a fresh one per run, because verl
#                     resumes from any checkpoint it finds there
#   CKPT_DIR          checkpoint directory, default $OUTPUT_DIR
#   SAVE_FREQ         save every N steps (and at the last step); default only the last step. sft saves its last
#                     step in any case.
#   ACTOR_CHECKPOINT_SAVE_CONTENTS   default [hf_model]: a HuggingFace export only (fp32 tensors, about 6.9 GB
#                     at 1.7B), which evaluation and the next stage read; it cannot resume training
#   USE_FUSED_KERNELS True (default) computes log-probs and entropy in chunks instead of materializing the full
#                     logits: about 27% less training time per 1k tokens and peak reserved memory 75 -> 39 GB at
#                     1.7B with 16k responses, with unchanged numerics. sft ignores it.
#   RAY_NUM_CPUS      CPUs Ray may use, default $SLURM_CPUS_ON_NODE or nproc
#   WANDB_MODE        default offline (runs land in $OUTPUT_DIR/wandb; upload later with `wandb sync`)
#   WANDB_PROJECT / WANDB_TAGS   default scaffold-opd / <method>,<student>; WANDB_ENTITY is passed through if set
#   EXTRA_HYDRA_ARGS  extra Hydra overrides split on spaces, e.g. data.seed=43
set -eo pipefail

# shellcheck source=pipeline/common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
LOG_TAG=train
setup_env
set -u
cd "$REPO_ROOT"

METHOD=${METHOD:-takeover}
case "$METHOD" in
    takeover | relay)
        METHOD=takeover
        LAUNCHER=opd/scripts/relay_opd/train.sh
        default_steps=40
        ;;
    opd)
        LAUNCHER=opd/scripts/baselines/opd.sh
        default_steps=40
        ;;
    grpo)
        LAUNCHER=opd/scripts/baselines/grpo.sh
        default_steps=60
        ;;
    sft)
        LAUNCHER=opd/scripts/baselines/sft.sh
        default_steps=0
        ;;
    *) die "METHOD must be takeover, opd, grpo or sft, got $METHOD" ;;
esac
distill=0
[[ "$METHOD" == takeover || "$METHOD" == opd ]] && distill=1
if [[ "${CONCLUDE_TRIGGER:-0}" == 1 ]]; then
    [[ "$METHOD" == takeover ]] || die "CONCLUDE_TRIGGER=1 applies to METHOD=takeover only"
    # The 45 default reflection triggers of opd/patches/vllm/speculative_decode.py, then the header words. With
    # " Final" in the list, the patch also counts "✅" right after "###" (RELAY_OPD_HEADER_CONCLUDE=auto).
    reflection="Wait, Wait,wait, wait,But, But,but, but,Hmm, Hmm, hmm,Actually, Actually,actually, actually,"
    reflection+="Hold, Hold,hold, hold,However, However,however, however,Yet, Yet,yet, yet,Oh, Oh,oh, oh,"
    reflection+="Alternatively, Alternatively,No, No,no, no,Ah, Ah,ah, ah,Oops, Oops,Well, Well"
    export RELAY_OPD_REFLECTION_TOKENS="${RELAY_OPD_REFLECTION_TOKENS:-$reflection},Final, Final,Answer, Answer"
fi

steps=${STEPS:-$default_steps}
batch=${BATCH:-128}
response_len=${RESPONSE_LEN:-16384}
[[ "$steps" =~ ^[0-9]+$ && "$batch" =~ ^[1-9][0-9]*$ && "$response_len" =~ ^[1-9][0-9]*$ ]] ||
    die "STEPS, BATCH and RESPONSE_LEN must be integers (got '$steps', '$batch', '$response_len')"
[[ "$METHOD" == sft ]] || ((steps > 0)) || die "METHOD=$METHOD needs STEPS > 0"

export STUDENT_MODEL=${STUDENT_MODEL:-$DEFAULT_STUDENT}
export TEACHER_MODEL=${TEACHER_MODEL:-$DEFAULT_TEACHER}
if [[ "$METHOD" == sft ]]; then
    export TRAIN_DATA=${TRAIN_DATA:?METHOD=sft needs TRAIN_DATA, the parquet written by pipeline/filter_teacher_data.py}
else
    export TRAIN_DATA=${TRAIN_DATA:-$DEFAULT_TRAIN_DATA}
fi
export BENCH=${BENCH:-$DEFAULT_BENCH_DIR}
export OUTPUT_DIR=${OUTPUT_DIR:-$OUTPUT_ROOT/${METHOD}_s${steps}_b${batch}_$(date +%Y%m%d_%H%M%S)}
export CKPT_DIR=${CKPT_DIR:-$OUTPUT_DIR}
export WANDB_DIR=$OUTPUT_DIR WANDB_MODE=${WANDB_MODE:-offline}
student_name=$(basename "$STUDENT_MODEL" | tr '[:upper:]' '[:lower:]')
export WANDB_TAGS=${WANDB_TAGS:-$METHOD,$student_name}

require_path "$STUDENT_MODEL/config.json"
((distill)) && require_path "$TEACHER_MODEL/config.json"
require_path "$TRAIN_DATA"
if [[ "$METHOD" != sft ]]; then
    # The launchers always load these two files, even with validation switched off.
    require_path "$BENCH/aime-24_verl.parquet"
    require_path "$BENCH/aime-2025_verl.parquet"
fi

n_gpus=$(gpu_count)
export N_GPUS=$n_gpus NUM_GPUS=$n_gpus
if ((distill)); then
    ((n_gpus >= 2)) || die "METHOD=$METHOD needs at least 2 GPUs (student and teacher); CUDA_VISIBLE_DEVICES lists $n_gpus"
    export ACTOR_GPUS_PER_NODE=${ACTOR_GPUS_PER_NODE:-$((n_gpus / 2))}
    export TEACHER_GPUS_PER_NODE=${TEACHER_GPUS_PER_NODE:-$((n_gpus - ACTOR_GPUS_PER_NODE))}
    ((ACTOR_GPUS_PER_NODE + TEACHER_GPUS_PER_NODE <= n_gpus)) ||
        die "ACTOR_GPUS_PER_NODE + TEACHER_GPUS_PER_NODE exceeds the $n_gpus visible GPUs"
fi
export TRAIN_BATCH_SIZE=$batch PPO_MINI_BATCH_SIZE=$batch
export MAX_RESPONSE_LENGTH=$response_len VAL_MAX_RESPONSE_LENGTH=$response_len
export ROLLOUT_N=${ROLLOUT_N:-8}
if [[ "$METHOD" == takeover ]]; then
    # The student's GPU hosts both the vLLM teacher (target) and the student (draft) during rollout. FSDP only
    # needs the card while vLLM sleeps, so vLLM gets a larger KV cache than the launcher's default.
    export ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.6}
fi
export ACTOR_CHECKPOINT_SAVE_CONTENTS=${ACTOR_CHECKPOINT_SAVE_CONTENTS:-[hf_model]}
export CHECKPOINT_SAVE_CONTENTS=$ACTOR_CHECKPOINT_SAVE_CONTENTS
if [[ "$METHOD" == sft ]]; then
    export SAVE_FREQ=${SAVE_FREQ:--1}
else
    export SAVE_FREQ=${SAVE_FREQ:-$steps}
fi
# No in-training validation: it would sample through the teacher-assisted rollout. Evaluate the student alone
# with pipeline/eval.sh instead.
export TEST_FREQ=${TEST_FREQ:--1} VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}

log "method=$METHOD steps=$steps batch=$batch response_len=$response_len save_freq=$SAVE_FREQ gpus=${CUDA_VISIBLE_DEVICES:-unset}"
log "student=$STUDENT_MODEL data=$TRAIN_DATA output=$OUTPUT_DIR checkpoints=$CKPT_DIR"
show_env STUDENT_MODEL TEACHER_MODEL TRAIN_DATA BENCH OUTPUT_DIR CKPT_DIR N_GPUS ACTOR_GPUS_PER_NODE \
    TEACHER_GPUS_PER_NODE TRAIN_BATCH_SIZE MAX_RESPONSE_LENGTH ROLLOUT_N SAVE_FREQ ACTOR_CHECKPOINT_SAVE_CONTENTS \
    RELAY_OPD_REFLECTION_TOKENS

# shellcheck disable=SC2206  # EXTRA_HYDRA_ARGS is a space-separated list on purpose
extra=(${EXTRA_HYDRA_ARGS:-})
# Later duplicates of a Hydra key win, so these override the launcher defaults (logger and project name).
common=('trainer.logger=[console,wandb]' "trainer.project_name=${WANDB_PROJECT:-scaffold-opd}"
    "trainer.default_local_dir=$CKPT_DIR")
if [[ "$METHOD" == sft ]]; then
    # The SFT launcher's default sequence parallelism needs flash-attn; 1 runs plain data parallel.
    export SP_SIZE=${SP_SIZE:-1}
    if ((steps > 0)); then
        export TOTAL_TRAINING_STEPS=$steps
    fi
    run bash "$LAUNCHER" "${common[@]}" ${extra[@]+"${extra[@]}"} "$@"
else
    run bash "$LAUNCHER" \
        "trainer.total_training_steps=$steps" \
        "data.train_max_samples=$((batch * steps * 2))" \
        "${common[@]}" \
        "trainer.rollout_data_dir=$OUTPUT_DIR/rollout_dump" \
        "ray_kwargs.ray_init.num_cpus=${RAY_NUM_CPUS:-${SLURM_CPUS_ON_NODE:-$(nproc)}}" \
        "actor_rollout_ref.model.use_fused_kernels=${USE_FUSED_KERNELS:-True}" \
        ${extra[@]+"${extra[@]}"} "$@"
fi
is_dry_run || log "done. Offline W&B run: $OUTPUT_DIR/wandb (upload with: wandb sync $OUTPUT_DIR/wandb/offline-run-*)"
