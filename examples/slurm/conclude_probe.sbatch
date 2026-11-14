#!/usr/bin/env bash
#SBATCH --job-name=scaffold-opd-conclude-probe
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%j.out
# Copyright 2026 Yi Zhang
# SPDX-License-Identifier: Apache-2.0
#
# Slurm wrapper around pipeline/conclude_probe.sh. Needs STUDENT; inference only.
# Submit from the repository root and add your cluster's partition and account on the command line:
#   sbatch --partition=<partition> --account=<account> examples/slurm/conclude_probe.sbatch
# Every variable the wrapped script reads passes through, e.g. MODEL_DIR, DATA_DIR, OUTPUT_ROOT, CKPT_DIR, VENV and
# ENV_SETUP (a file to source first for module loads or cache locations). Compute nodes without internet are fine:
# models and data are read from local paths, HF_HUB_OFFLINE=1 and WANDB_MODE=offline are the defaults.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
exec bash pipeline/conclude_probe.sh "$@"
