#!/usr/bin/env python3
"""Published ToolUse metrics with CLI discovery and model-card macro F1."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re


NO_TOOL = "no_tool"
ERROR_LABEL = "__rollout_error__"
TOOL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
UPSTREAM_COMMIT = "73772633663dfe02eff558a85eacbac9f617d329"


def predicted(record):
    if record.get("error") is not None:
        return ERROR_LABEL
    message = (record.get("completion") or [{}])[-1]
    for tool_call in message.get("tool_calls") or []:
        name = tool_call.get("name") or tool_call.get("function", {}).get("name")
        if name:
            return name
    match = TOOL_RE.search(message.get("content") or "")
    if match:
        try:
            return json.loads(match.group(1)).get("name") or NO_TOOL
        except json.JSONDecodeError:
            pass
    return NO_TOOL


def expected(record):
    answer = str(record.get("answer") or "").strip()
    if answer.lower() in {"", "none", "negative", "negatives", "no_tool"}:
        return NO_TOOL
    return answer


def load(path):
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def group_rollouts(records, k):
    if not records:
        return []
    for key in (
        "example_id",
        "id",
        "sample_id",
        "prompt_id",
        "question_id",
        "task_id",
    ):
        if key in records[0]:
            order = []
            buckets = defaultdict(list)
            for record in records:
                if record[key] not in buckets:
                    order.append(record[key])
                buckets[record[key]].append(record)
            return [buckets[item] for item in order]
    if "prompt" in records[0]:
        order = []
        buckets = defaultdict(list)
        for record in records:
            key = record["prompt"]
            if not isinstance(key, str):
                key = json.dumps(key, sort_keys=True)
            if key not in buckets:
                order.append(key)
            buckets[key].append(record)
        return [buckets[item] for item in order]
    return [records[index : index + k] for index in range(0, len(records), k)]


def prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return precision, recall, f1


def pass1_per_class(groups, labels):
    output = {}
    for label in labels:
        tp = fp = fn = 0
        for group in groups:
            for record in group:
                actual = expected(record)
                prediction = predicted(record)
                if actual == label and prediction == label:
                    tp += 1
                elif actual != label and prediction == label:
                    fp += 1
                elif actual == label and prediction != label:
                    fn += 1
        output[label] = prf(tp, fp, fn) + (tp + fn,)
    return output


def passk_per_class(groups, labels):
    output = {}
    for label in labels:
        tp = fp = fn = 0
        for group in groups:
            actual = expected(group[0])
            predictions = {predicted(record) for record in group}
            predicted_label = label in predictions
            if actual == label and predicted_label:
                tp += 1
            elif actual != label and predicted_label:
                fp += 1
            elif actual == label and not predicted_label:
                fn += 1
        output[label] = prf(tp, fp, fn) + (tp + fn,)
    return output


def pass1_trigger(groups):
    tp = fp = fn = tn = 0
    for group in groups:
        for record in group:
            actual = expected(record) != NO_TOOL
            prediction_label = predicted(record)
            # A rollout error is always an incorrect trigger decision.
            prediction = (
                not actual
                if prediction_label == ERROR_LABEL
                else prediction_label != NO_TOOL
            )
            if actual and prediction:
                tp += 1
            elif not actual and prediction:
                fp += 1
            elif actual and not prediction:
                fn += 1
            else:
                tn += 1
    precision, recall, f1 = prf(tp, fp, fn)
    n = tp + fp + fn + tn
    return {
        "P": precision,
        "R": recall,
        "F1": f1,
        "acc": (tp + tn) / n if n else 0.0,
    }


def passk_trigger(groups):
    tp = fp = fn = tn = 0
    for group in groups:
        actual = expected(group[0]) != NO_TOOL
        valid_predictions = [
            predicted(record)
            for record in group
            if predicted(record) != ERROR_LABEL
        ]
        # Valid rollouts can still satisfy pass@k. If every rollout failed,
        # force an incorrect trigger decision.
        prediction = (
            any(label != NO_TOOL for label in valid_predictions)
            if valid_predictions
            else not actual
        )
        if actual and prediction:
            tp += 1
        elif not actual and prediction:
            fp += 1
        elif actual and not prediction:
            fn += 1
        else:
            tn += 1
    precision, recall, f1 = prf(tp, fp, fn)
    n = tp + fp + fn + tn
    return {
        "P": precision,
        "R": recall,
        "F1": f1,
        "acc": (tp + tn) / n if n else 0.0,
    }


def pass1_accuracy(groups):
    n = hits = 0
    for group in groups:
        for record in group:
            n += 1
            hits += int(expected(record) == predicted(record))
    return hits / n if n else 0.0


def passk_accuracy(groups):
    n = hits = 0
    for group in groups:
        n += 1
        predictions = {predicted(record) for record in group}
        hits += int(expected(group[0]) in predictions)
    return hits / n if n else 0.0


def latest_results_path(results_root, run_name):
    eval_root = (
        results_root
        / run_name
        / "vf"
        / "evals"
        / "med_app_env--{}".format(run_name)
    )
    candidates = list(eval_root.rglob("results.jsonl"))
    if not candidates:
        raise FileNotFoundError("no results.jsonl under {}".format(eval_root))
    return max(candidates, key=lambda path: path.stat().st_mtime)


def macro_f1(per_class, labels):
    values = [per_class[label][2] for label in labels]
    return sum(values) / len(values) if values else 0.0


def build_report(path, k, expected_samples):
    records = load(path)
    errors = sum(record.get("error") is not None for record in records)

    groups = group_rollouts(records, k)
    if len(groups) != expected_samples:
        raise ValueError(
            "{}: expected {} samples, found {}".format(
                path,
                expected_samples,
                len(groups),
            )
        )
    wrong_group_sizes = sum(len(group) != k for group in groups)
    if wrong_group_sizes:
        raise ValueError(
            "{}: {} sample groups do not contain exactly {} rollouts".format(
                path,
                wrong_group_sizes,
                k,
            )
        )

    expected_labels = sorted(
        {expected(record) for group in groups for record in group}
    )
    predicted_labels = {
        predicted(record) for group in groups for record in group
    }
    predicted_labels.discard(ERROR_LABEL)
    all_labels = sorted(set(expected_labels) | predicted_labels)
    pass1_classes = pass1_per_class(groups, all_labels)
    passk_classes = passk_per_class(groups, all_labels)
    pass1 = {
        "accuracy": pass1_accuracy(groups),
        "trigger": pass1_trigger(groups),
        "per_class": {
            label: {
                "precision": values[0],
                "recall": values[1],
                "f1": values[2],
                "support": values[3],
            }
            for label, values in pass1_classes.items()
        },
        "macro_f1_expected_labels": macro_f1(
            pass1_classes,
            expected_labels,
        ),
        "macro_f1_all_labels": macro_f1(pass1_classes, all_labels),
    }
    passk = {
        "accuracy": passk_accuracy(groups),
        "trigger": passk_trigger(groups),
        "per_class": {
            label: {
                "precision": values[0],
                "recall": values[1],
                "f1": values[2],
                "support": values[3],
            }
            for label, values in passk_classes.items()
        },
        "macro_f1_expected_labels": macro_f1(
            passk_classes,
            expected_labels,
        ),
        "macro_f1_all_labels": macro_f1(passk_classes, all_labels),
    }
    return {
        "input": str(path),
        "n_samples": len(groups),
        "n_rollouts": len(records),
        "n_errors": errors,
        "expected_labels": expected_labels,
        "all_labels": all_labels,
        "pass@1": pass1,
        "pass@{}".format(k): passk,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--expected-samples", type=int, default=5122)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    reports = {}
    for run_name in args.runs:
        path = latest_results_path(args.results_root, run_name)
        reports[run_name] = build_report(
            path,
            args.k,
            args.expected_samples,
        )

    print(
        "run | rollouts | errors | pass@1 accuracy | pass@1 trigger F1 | "
        "pass@1 macro F1 | pass@{} accuracy | pass@{} trigger F1 | "
        "pass@{} macro F1".format(args.k, args.k, args.k)
    )
    print(
        "--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---:"
    )
    for run_name in args.runs:
        report = reports[run_name]
        pass1 = report["pass@1"]
        passk = report["pass@{}".format(args.k)]
        print(
            "{} | {} | {} | {:.2f}% | {:.2f}% | {:.2f}% | "
            "{:.2f}% | {:.2f}% | {:.2f}%".format(
                run_name,
                report["n_rollouts"],
                report["n_errors"],
                pass1["accuracy"] * 100,
                pass1["trigger"]["F1"] * 100,
                pass1["macro_f1_expected_labels"] * 100,
                passk["accuracy"] * 100,
                passk["trigger"]["F1"] * 100,
                passk["macro_f1_expected_labels"] * 100,
            )
        )

    if args.output_json:
        payload = {
            "protocol": (
                "vistec-AI/thaillm-medical-post-training@{}".format(
                    UPSTREAM_COMMIT
                )
            ),
            "k": args.k,
            "models": reports,
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print("Wrote {}".format(args.output_json))


if __name__ == "__main__":
    main()
