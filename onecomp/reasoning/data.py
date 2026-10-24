# Copyright 2025-2026 Fujitsu Ltd.

"""Versioned benchmark import and deliberately conservative answer grading."""

import hashlib
import json
import random
import re
from fractions import Fraction
from pathlib import Path

DATASETS = {
    "math500": ("HuggingFaceH4/MATH-500", None, "test"),
    "amc2023": ("math-ai/amc23", None, "test"),
    "aime2024": ("HuggingFaceH4/aime_2024", None, "train"),
    "gpqa_diamond": ("Idavidrein/gpqa", "gpqa_diamond", "train"),
}
PROMPT_TEMPLATE = (
    "Solve the following problem. Put only your final answer inside \\boxed{{}}.\n\n{question}"
)


def extract_answer(text):
    """Last balanced boxed answer, then explicit answer line or one-line output.

    Never extract an arbitrary last number from unfinished reasoning. This is an
    exact/numeric grader, not a full symbolic equivalence engine for MATH-500.
    """
    starts = list(re.finditer(r"\\boxed\s*\{", text))
    for match in reversed(starts):
        depth = 1
        for end in range(match.end(), len(text)):
            if text[end] == "{":
                depth += 1
            elif text[end] == "}":
                depth -= 1
            if depth == 0:
                return text[match.end() : end].strip()
    match = re.search(r"(?:final\s+answer|answer)\s*[:=]\s*([^\n]+)", text, re.I)
    if match:
        return match.group(1).strip().rstrip(".")
    stripped = text.strip()
    return stripped.rstrip(".") if "\n" not in stripped and len(stripped) <= 120 else ""


def normalize_answer(answer):
    answer = str(answer).strip().replace("$", "").replace(r"\left", "").replace(r"\right", "")
    answer = re.sub(r"\\(?:text|mathrm)\{([^{}]*)\}", r"\1", answer)
    answer = re.sub(r"\\(?:dfrac|tfrac|frac)\{(-?\d+)\}\{(-?\d+)\}", r"\1/\2", answer)
    answer = re.sub(r"\s+", "", answer)
    if re.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?", answer):
        answer = answer.replace(",", "")
    if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:/[+-]?\d+)?", answer):
        try:
            return str(Fraction(answer))
        except (ValueError, ZeroDivisionError):
            pass
    return answer


def grade_answer(text, expected):
    actual = extract_answer(text)
    return bool(actual) and normalize_answer(actual) == normalize_answer(expected)


def read_jsonl(path):
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def load_samples(path, split, limit=None):
    rows = read_jsonl(path)
    ids = set()
    for row in rows:
        for field in ("sample_id", "dataset", "question", "answer", "split", "dataset_revision"):
            if field not in row:
                raise ValueError(f"Dataset row missing {field}")
        key = (row["dataset"], str(row["sample_id"]))
        if key in ids:
            raise ValueError(f"Duplicate sample ID {key}")
        ids.add(key)
    rows = [row for row in rows if row["split"] == split]
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        rows = rows[:limit]
    if not rows:
        raise ValueError(f"No examples for split {split!r}")
    if len({(row["dataset"], row["dataset_revision"]) for row in rows}) != 1:
        raise ValueError("One dataset/revision per run is required; aggregate suites offline")
    return rows


def prepare_dataset(name, output, *, revision, seed=0, cache_dir=None):
    """Download a declared revision and freeze disjoint 20/20/60% split roles.

    AMC23 here is math-ai's 40-question subset, not the 83-row mixed-year AMC set.
    GPQA uses a seeded choice permutation, frozen identically for every policy.
    Data access failures propagate; gated datasets are not silently substituted.
    """
    from datasets import load_dataset

    if not revision or not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        raise ValueError("Specify an immutable dataset revision (Hub commit SHA)")
    repo, subset, source_split = DATASETS[name]
    dataset = load_dataset(
        repo, name=subset, split=source_split, revision=revision, cache_dir=cache_dir
    )
    rows = []
    for index, item in enumerate(dataset):
        question = item.get("problem", item.get("question", item.get("Question")))
        identifier = str(
            item.get(
                "unique_id", item.get("id", hashlib.sha256(question.encode()).hexdigest()[:16])
            )
        )
        if name == "gpqa_diamond":
            choices = [item["Correct Answer"]] + [
                item[f"Incorrect Answer {i}"] for i in range(1, 4)
            ]
            order = list(range(4))
            random.Random(f"{seed}:{identifier}").shuffle(order)
            question += "\n" + "\n".join(f"{chr(65+i)}. {choices[j]}" for i, j in enumerate(order))
            question += (
                "\nReturn the uppercase option letter (A, B, C, or D) inside the final answer box."
            )
            answer = chr(65 + order.index(0))
        else:
            answer = str(item["answer"])
        rows.append(
            {
                "sample_id": identifier,
                "dataset": name,
                "dataset_revision": revision,
                "question": question,
                "answer": answer,
                "difficulty": str(item.get("level", "unknown")),
                "source": repo,
                "source_split": source_split,
            }
        )
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    n_dev = max(1, len(rows) // 5)
    for rank, index in enumerate(order):
        rows[index]["split"] = (
            "development" if rank < n_dev else "calibration" if rank < 2 * n_dev else "evaluation"
        )
    write_jsonl(output, rows)
    return {
        "dataset": name,
        "revision": revision,
        "rows": len(rows),
        "seed": seed,
        "source": repo,
        "split_strategy": "seeded_disjoint_20_20_60",
        "sha256": hashlib.sha256(Path(output).read_bytes()).hexdigest(),
    }
