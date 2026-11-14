# Copyright 2026 Yi Zhang
# SPDX-License-Identifier: Apache-2.0
"""Collect evaluation and training results into one JSON (the numbers behind the README tables and figures).

    python3 tools/analysis/collect_results.py \
        --eval base=outputs/eval/base \
        --eval scaffold_40=outputs/compare_<id>/eval/scaffold_step40 \
        --eval opd_40=outputs/compare_<id>/eval/opd_step40 \
        --pair scaffold_40:opd_40 --pair scaffold_40:base --pair scaffold_40:opd_40@16384 \
        --budget scaffold_40:opd_40 \
        --train_log scaffold=outputs/compare_<id>/scaffold/training.log \
        > results.json

Each --eval directory holds <bench>.jsonl from opd/eval/math_benchmarks.py, one record per sample.

- Accuracy is avg@k: samples are averaged within a problem first, then over problems; the 4-benchmark average is
  the plain mean of the benchmark accuracies.
- A paired difference A:B averages the per-problem differences between A and B. Its standard error is the standard
  deviation of those differences over sqrt(#problems); for the 4-benchmark average the benchmarks are treated as
  independent, SE = sqrt(sum of squared benchmark SEs) / 4. "A:B@N" evaluates the pair under an N-token budget.
- Under a budget of N tokens a sample counts as correct only if it stopped (EOS) within N tokens and was graded
  correct. Decoding is autoregressive, so the first N tokens of a 32k sample are exactly what an N-token run would
  have produced; the only approximation is that an answer written before a cut that never reached EOS counts wrong.
- --train_log parses verl's console lines ("step:N - key:value - ...") from a run's training.log.

Standard library only, so it runs with any Python 3.10+.
"""

import argparse
import collections
import json
import math
import re
import statistics as st

DEFAULT_BENCHES = "math500,aime24,aime25,aime26"
# 1k to 32k in quarter-octave steps, so budget curves are smooth on a log2 axis.
BUDGETS = sorted({round(1024 * 2 ** (i / 4)) for i in range(21)})
TRAIN_KEYS = [
    "opd/k1/student_leg_mean",
    "relay/teacher_token_ratio",
    "relay/teacher_segments_per_resp_mean",
    "relay/teacher_segment_len_mean",
    "relay/resp_with_teacher_ratio",
    "response_length/mean",
    "response_length/clip_ratio",
    "timing_s/step",
]
NUMPY_SCALAR = re.compile(r"np\.(?:float|int)\d*\(([^)]*)\)")


def named_path(text: str) -> tuple[str, str]:
    name, sep, path = text.partition("=")
    if not sep or not name or not path:
        raise argparse.ArgumentTypeError(f"expected NAME=PATH, got {text!r}")
    return name, path


def pair_spec(text: str) -> tuple[str, str, int | None]:
    match = re.fullmatch(r"([^:@]+):([^:@]+)(?:@(\d+))?", text)
    if not match:
        raise argparse.ArgumentTypeError(f"expected A:B or A:B@BUDGET, got {text!r}")
    a, b, budget = match.groups()
    return a, b, int(budget) if budget else None


def load(path: str, benches: list[str]) -> dict:
    """bench -> problem_idx -> [(correct, gen_len, stopped)]"""
    out = {}
    for bench in benches:
        per = collections.defaultdict(list)
        with open(f"{path}/{bench}.jsonl") as f:
            for line in f:
                d = json.loads(line)
                per[d["problem_idx"]].append((bool(d["correct"]), int(d["gen_len"]), d.get("finish_reason") == "stop"))
        out[bench] = per
    return out


def per_problem(samples: dict, budget: int | None = None) -> dict:
    """Per-problem accuracy in percent; budget=None is the plain score."""
    if budget is None:
        return {k: 100 * sum(c for c, _, _ in v) / len(v) for k, v in samples.items()}
    return {k: 100 * sum(c and s and n <= budget for c, n, s in v) / len(v) for k, v in samples.items()}


def average(data: dict, benches: list[str], budget: int | None = None) -> float:
    return st.mean(st.mean(per_problem(data[b], budget).values()) for b in benches)


def paired(d1: dict, d2: dict, benches: list[str], budget: int | None = None) -> tuple[float, float]:
    total, var = 0.0, 0.0
    for b in benches:
        p1, p2 = per_problem(d1[b], budget), per_problem(d2[b], budget)
        if p1.keys() != p2.keys():
            raise ValueError(f"{b}: the two runs evaluated different problems")
        diffs = [p1[k] - p2[k] for k in p1]
        total += st.mean(diffs)
        var += (st.stdev(diffs) / math.sqrt(len(diffs))) ** 2
    return total / len(benches), math.sqrt(var) / len(benches)


def finished(samples: dict, budget: int) -> float:
    xs = [x for v in samples.values() for x in v]
    return 100 * sum(s and n <= budget for _, n, s in xs) / len(xs)


def parse_log(path: str) -> dict:
    """verl console lines `step:N - key:value - ...`; the last line for a step wins."""
    rows = {}
    with open(path, errors="replace") as f:
        for line in f:
            m = re.search(r"step:(\d+) - (.*)", line)
            if not m:
                continue
            values = {}
            for part in NUMPY_SCALAR.sub(r"\1", m.group(2)).split(" - "):
                key, _, value = part.partition(":")
                if key.strip() in TRAIN_KEYS:
                    try:
                        values[key.strip()] = float(value.split()[0])
                    except (ValueError, IndexError):
                        pass
            if values:
                rows[int(m.group(1))] = values
    steps = sorted(rows)
    return {"step": steps, **{k: [rows[s].get(k) for s in steps] for k in TRAIN_KEYS}}


def rounded(x):
    if isinstance(x, float):
        return round(x, 4)
    if isinstance(x, dict):
        return {k: rounded(v) for k, v in x.items()}
    if isinstance(x, list):
        return [rounded(v) for v in x]
    return x


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval", type=named_path, action="append", required=True, metavar="NAME=DIR")
    parser.add_argument("--pair", type=pair_spec, action="append", default=[], metavar="A:B[@BUDGET]")
    parser.add_argument("--budget", type=pair_spec, default=None, metavar="A:B", help="budget curves for A and B")
    parser.add_argument("--train_log", type=named_path, action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--benches", default=DEFAULT_BENCHES)
    args = parser.parse_args()
    benches = args.benches.split(",")

    names = [name for name, _ in args.eval]
    if len(set(names)) != len(names):
        parser.error("--eval names must be unique")
    wanted = [(a, b) for a, b, _ in args.pair] + ([args.budget[:2]] if args.budget else [])
    for a, b in wanted:
        for name in (a, b):
            if name not in names:
                parser.error(f"{name!r} is used in --pair/--budget but has no --eval")

    data = {name: load(path, benches) for name, path in args.eval}
    evals = {}
    for name, d in data.items():
        aime = [b for b in benches if b.startswith("aime")]
        aime_lens = [n for b in aime for v in d[b].values() for _, n, _ in v]
        math_lens = [n for b in benches if b == "math500" for v in d[b].values() for _, n, _ in v]
        evals[name] = {
            "acc": {b: st.mean(per_problem(d[b]).values()) for b in benches},
            "avg4": average(d, benches),
            "aime_len_mean": st.mean(aime_lens) if aime_lens else None,
            "math_len_mean": st.mean(math_lens) if math_lens else None,
            "samples_per_problem": {b: round(st.mean(len(v) for v in d[b].values()), 2) for b in benches},
        }

    pairs = {}
    for a, b, budget in args.pair:
        key = f"{a}-{b}" + (f"@{budget}" if budget else "")
        pairs[key] = dict(zip(("diff", "se"), paired(data[a], data[b], benches, budget), strict=True))

    result = {"benches": benches, "eval": evals, "paired": pairs}
    if args.budget:
        a, b, _ = args.budget
        budget = {"budgets": BUDGETS}
        for name in (a, b):
            d = data[name]
            budget[name] = {
                "avg4": [average(d, benches, x) for x in BUDGETS],
                "aime_finished": [st.mean(finished(d[k], x) for k in benches if k.startswith("aime")) for x in BUDGETS],
                "math_finished": [finished(d["math500"], x) for x in BUDGETS] if "math500" in benches else [],
            }
        diffs = [paired(data[a], data[b], benches, x) for x in BUDGETS]
        budget[f"{a}-{b}"] = {"diff": [d for d, _ in diffs], "se": [s for _, s in diffs]}
        result["budget"] = budget
    if args.train_log:
        result["train"] = {name: parse_log(path) for name, path in args.train_log}

    print(json.dumps(rounded(result), ensure_ascii=False))


if __name__ == "__main__":
    main()
