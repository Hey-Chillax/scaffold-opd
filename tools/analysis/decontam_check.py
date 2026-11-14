# Copyright 2026 Yi Zhang
# SPDX-License-Identifier: Apache-2.0
"""Report how much of the training set overlaps the evaluation benchmarks.

Read-only and CPU-only. Run it from the repo root with the project venv:

    python tools/analysis/decontam_check.py [--train <parquet>] [--bench_dir <dir>]

A benchmark problem counts as a near match when at least --threshold of its word 8-grams also occur in
one training problem. A near match with the same answer is a real leak; with a different answer it is
usually the same template with other numbers.
"""

import argparse
import collections
import os
import re

import pandas as pd

BENCHES = {
    "math500": "math500.parquet",
    "aime24": "aime-24.parquet",
    "aime25": "aime-2025.parquet",
    "aime26": "aime26.parquet",
}
NGRAM = 8


def problem_text(prompt) -> str:
    for message in prompt:
        if message["role"] == "user":
            return str(message["content"])
    raise ValueError("row has no user message")


def words(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z]+|\d+", text.lower()))


def ngrams(tokens: tuple[str, ...]) -> set[tuple[str, ...]]:
    return {tokens[i : i + NGRAM] for i in range(len(tokens) - NGRAM + 1)}


def answer(row) -> str:
    return re.sub(r"\s+", "", str(row["reward_model"]["ground_truth"]))


def main() -> None:
    data_dir = os.environ.get("DATA_DIR", "data")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", default=f"{data_dir}/train/dapo_math_17k_en.parquet")
    parser.add_argument("--bench_dir", default=f"{data_dir}/bench")
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--examples", type=int, default=3)
    args = parser.parse_args()

    train = pd.read_parquet(args.train)
    texts = [problem_text(prompt) for prompt in train["prompt"]]
    answers = [answer(row) for _, row in train.iterrows()]
    tokens = [words(text) for text in texts]
    exact = {t: i for i, t in enumerate(tokens)}
    index: dict[tuple[str, ...], list[int]] = collections.defaultdict(list)
    for i, t in enumerate(tokens):
        for gram in ngrams(t):
            index[gram].append(i)
    print(f"train problems: {len(texts)}, distinct 8-grams: {len(index)}")

    for bench, filename in BENCHES.items():
        frame = pd.read_parquet(os.path.join(args.bench_dir, filename))
        n_exact = n_near = n_leak = 0
        examples = []
        for _, row in frame.iterrows():
            text = problem_text(row["prompt"])
            t = words(text)
            gold = answer(row)
            if t in exact:
                n_exact += 1
                n_leak += answers[exact[t]] == gold
                if len(examples) < args.examples:
                    examples.append(("exact", text, answers[exact[t]], gold))
                continue
            grams = ngrams(t)
            if not grams:
                continue
            hits = collections.Counter(i for gram in grams for i in index.get(gram, ()))
            if not hits:
                continue
            best, shared = hits.most_common(1)[0]
            if shared / len(grams) >= args.threshold:
                n_near += 1
                n_leak += answers[best] == gold
                if len(examples) < args.examples:
                    examples.append((f"near {shared / len(grams):.2f}", text, answers[best], gold))
        total = len(frame)
        print(
            f"{bench:8s} problems={total:3d} exact={n_exact:3d} near={n_near:3d} "
            f"same_answer={n_leak:3d} ({100 * n_leak / total:.1f}%)"
        )
        for kind, text, train_answer, gold in examples:
            print(f"    [{kind}] train answer {train_answer!r}, benchmark answer {gold!r}: {text[:90]!r}")


if __name__ == "__main__":
    main()
