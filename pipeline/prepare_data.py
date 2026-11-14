# Copyright 2026 Yi Zhang
# SPDX-License-Identifier: Apache-2.0
"""Convert the raw math datasets into the parquet layout the pipeline reads.

    python pipeline/prepare_data.py --raw_dir data/raw --out_dir data [--tokenizer models/Qwen3-1.7B]

Inputs (downloaded beforehand, read in place and never modified; see the README for the download commands):
    <raw_dir>/DAPO-Math-17k-Processed/en/train-00000-of-00001.parquet   open-r1/DAPO-Math-17k-Processed, config "en"
    <raw_dir>/aime_2024/data/train-00000-of-00001.parquet               HuggingFaceH4/aime_2024
    <raw_dir>/AIME2025/aime2025-I.jsonl, aime2025-II.jsonl             opencompass/AIME2025
    <raw_dir>/aime_2026/data/train-00000-of-00001.parquet               MathArena/aime_2026
    <raw_dir>/MATH-500/test.jsonl                                       HuggingFaceH4/MATH-500
Each input can also be given directly with --dapo, --aime24, --aime25, --aime26 and --math500 (.parquet or .jsonl).

Every row uses the same prompt: a system message asking for step-by-step reasoning with the final answer in
\\boxed{}, then the problem as the user message. verl applies the chat template later, with the student's
tokenizer and enable_thinking=False.

Outputs (existing files are never overwritten; the script stops instead):
    <out_dir>/train/dapo_math_17k_en.parquet          TRAIN_DATA
    <out_dir>/bench/aime-24_verl.parquet              read by the training launchers even with validation off
    <out_dir>/bench/aime-2025_verl.parquet
    <out_dir>/bench/{aime-24,aime-2025,aime26,math500}.parquet   read by pipeline/eval.sh
"""

import argparse
import os

import pandas as pd

SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


def build_rows(problems, answers, data_source: str, split: str) -> list[dict]:
    rows = []
    for index, (problem, answer) in enumerate(zip(problems, answers, strict=True)):
        problem, answer = str(problem).strip(), str(answer).strip()
        if not problem or not answer:
            raise ValueError(f"{data_source} row {index} has an empty problem or answer")
        rows.append(
            {
                "data_source": data_source,
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": problem},
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {"split": split, "index": index},
            }
        )
    return rows


def write_parquet(rows: list[dict], path: str) -> None:
    if os.path.exists(path):
        raise FileExistsError(f"{path} already exists; move it away first if you want to regenerate it")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)
    print(f"wrote {len(rows):>6} rows -> {path}")


def read_table(*paths: str) -> pd.DataFrame:
    frames = []
    for path in paths:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{path} not found (see --help for the expected layout)")
        frames.append(pd.read_parquet(path) if path.endswith(".parquet") else pd.read_json(path, lines=True))
    return pd.concat(frames, ignore_index=True)


def column(frame: pd.DataFrame, *names: str) -> pd.Series:
    for name in names:
        if name in frame.columns:
            return frame[name]
    raise KeyError(f"none of the columns {names} found; columns are {list(frame.columns)}")


def dapo_answers(frame: pd.DataFrame) -> list[str]:
    # DAPO-Math-17k-Processed keeps the answer in reward_model.ground_truth (and in solution).
    return [
        rm["ground_truth"] if isinstance(rm, dict) else solution
        for rm, solution in zip(frame["reward_model"], frame["solution"], strict=True)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw_dir", default="data/raw", help="root of the raw downloads")
    parser.add_argument("--out_dir", default="data", help="DATA_DIR for the pipeline")
    parser.add_argument("--dapo", default=None)
    parser.add_argument("--aime24", default=None)
    parser.add_argument("--aime25", nargs="+", default=None)
    parser.add_argument("--aime26", default=None)
    parser.add_argument("--math500", default=None)
    parser.add_argument("--tokenizer", default=None, help="student model path; prints the rendered first prompt")
    args = parser.parse_args()
    raw = args.raw_dir

    dapo = read_table(args.dapo or f"{raw}/DAPO-Math-17k-Processed/en/train-00000-of-00001.parquet")
    aime24 = read_table(args.aime24 or f"{raw}/aime_2024/data/train-00000-of-00001.parquet")
    aime25 = read_table(*(args.aime25 or [f"{raw}/AIME2025/aime2025-I.jsonl", f"{raw}/AIME2025/aime2025-II.jsonl"]))
    aime26 = read_table(args.aime26 or f"{raw}/aime_2026/data/train-00000-of-00001.parquet")
    math500 = read_table(args.math500 or f"{raw}/MATH-500/test.jsonl")

    train_rows = build_rows(dapo["prompt"], dapo_answers(dapo), "math_dapo", "train")
    aime24_rows = build_rows(column(aime24, "problem"), column(aime24, "answer"), "aime24", "test")
    aime25_rows = build_rows(column(aime25, "question", "problem"), column(aime25, "answer"), "aime25", "test")
    aime26_rows = build_rows(column(aime26, "problem", "question"), column(aime26, "answer"), "aime26", "test")
    math500_rows = build_rows(column(math500, "problem"), column(math500, "answer"), "math500", "test")

    train_dir, bench_dir = os.path.join(args.out_dir, "train"), os.path.join(args.out_dir, "bench")
    write_parquet(train_rows, f"{train_dir}/dapo_math_17k_en.parquet")
    write_parquet(aime24_rows, f"{bench_dir}/aime-24_verl.parquet")
    write_parquet(aime25_rows, f"{bench_dir}/aime-2025_verl.parquet")
    write_parquet(aime24_rows, f"{bench_dir}/aime-24.parquet")
    write_parquet(aime25_rows, f"{bench_dir}/aime-2025.parquet")
    write_parquet(aime26_rows, f"{bench_dir}/aime26.parquet")
    write_parquet(math500_rows, f"{bench_dir}/math500.parquet")

    if args.tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        rendered = tokenizer.apply_chat_template(
            train_rows[0]["prompt"], tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        print("\n--- first training prompt as the student sees it ---")
        print(rendered)
        print(f"--- ground truth: {train_rows[0]['reward_model']['ground_truth']}")


if __name__ == "__main__":
    main()
