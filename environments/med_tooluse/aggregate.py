"""Tool-call accuracy, trigger metrics, and per-class P/R/F1 across runs."""

import argparse
import json, re
from pathlib import Path
from collections import defaultdict

NO_TOOL = "no_tool"
ERROR_LABEL = "__rollout_error__"
_TOOL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


# ── extraction ────────────────────────────────────────────────────────────────
def predicted(r: dict) -> str:
    if r.get("error") is not None:
        return ERROR_LABEL
    msg = (r.get("completion") or [{}])[-1]
    for tc in msg.get("tool_calls") or []:
        if name := (tc.get("name") or tc.get("function", {}).get("name")):
            return name
    if m := _TOOL_RE.search(msg.get("content") or ""):
        try:
            return json.loads(m.group(1)).get("name") or NO_TOOL
        except json.JSONDecodeError:
            pass
    return NO_TOOL


def expected(r: dict) -> str:
    ans = str(r.get("answer") or "").strip()
    return NO_TOOL if ans.lower() in {"", "none", "negative", "negatives", "no_tool"} else ans


def load(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    return json.loads(text) if text.startswith("[") else [json.loads(l) for l in text.splitlines() if l.strip()]


# ── group rollouts into samples ───────────────────────────────────────────────
def group_rollouts(records: list[dict], k: int) -> list[list[dict]]:
    """Prefer grouping by a stable id, then by prompt, else fall back to consecutive chunks of k."""
    if not records:
        return []
    for key in ("example_id", "id", "sample_id", "prompt_id", "question_id", "task_id"):
        if key in records[0]:
            order, buckets = [], defaultdict(list)
            for r in records:
                if r[key] not in buckets:
                    order.append(r[key])
                buckets[r[key]].append(r)
            return [buckets[i] for i in order]
    if "prompt" in records[0]:
        order, buckets = [], defaultdict(list)
        for r in records:
            key = r["prompt"] if isinstance(r["prompt"], str) else json.dumps(r["prompt"], sort_keys=True)
            if key not in buckets:
                order.append(key)
            buckets[key].append(r)
        return [buckets[i] for i in order]
    return [records[i : i + k] for i in range(0, len(records), k)]


# ── metrics ───────────────────────────────────────────────────────────────────
def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
    return pr, rc, f1


def pass1_per_class(groups, labels):
    out = {}
    for l in labels:
        tp = fp = fn = 0
        for g in groups:
            for r in g:
                e, p = expected(r), predicted(r)
                if e == l and p == l:
                    tp += 1
                elif e != l and p == l:
                    fp += 1
                elif e == l and p != l:
                    fn += 1
        out[l] = (*_prf(tp, fp, fn), tp + fn)
    return out


def passk_per_class(groups, labels):
    """Label C 'predicted' if ANY rollout predicts C."""
    out = {}
    for l in labels:
        tp = fp = fn = 0
        for g in groups:
            if not g:
                continue
            e, preds = expected(g[0]), {predicted(r) for r in g}
            pred_l = l in preds
            if e == l and pred_l:
                tp += 1
            elif e != l and pred_l:
                fp += 1
            elif e == l and not pred_l:
                fn += 1
        out[l] = (*_prf(tp, fp, fn), tp + fn)
    return out


def pass1_trigger(groups):
    tp = fp = fn = tn = 0
    for g in groups:
        for r in g:
            et = expected(r) != NO_TOOL
            # A failed rollout is always scored as the wrong trigger decision:
            # missed trigger for a tool case, false trigger for a no-tool case.
            pt = (not et) if predicted(r) == ERROR_LABEL else predicted(r) != NO_TOOL
            if et and pt:
                tp += 1
            elif not et and pt:
                fp += 1
            elif et and not pt:
                fn += 1
            else:
                tn += 1
    pr, rc, f1 = _prf(tp, fp, fn)
    n = tp + fp + fn + tn
    return {"P": pr, "R": rc, "F1": f1, "acc": (tp + tn) / n if n else 0.0}


def passk_trigger(groups):
    tp = fp = fn = tn = 0
    for g in groups:
        if not g:
            continue
        et = expected(g[0]) != NO_TOOL
        valid_predictions = [predicted(r) for r in g if predicted(r) != ERROR_LABEL]
        # pass@K can still succeed when at least one rollout is valid. If every
        # rollout failed, force an incorrect trigger decision.
        pt = (
            any(prediction != NO_TOOL for prediction in valid_predictions)
            if valid_predictions
            else not et
        )
        if et and pt:
            tp += 1
        elif not et and pt:
            fp += 1
        elif et and not pt:
            fn += 1
        else:
            tn += 1
    pr, rc, f1 = _prf(tp, fp, fn)
    n = tp + fp + fn + tn
    return {"P": pr, "R": rc, "F1": f1, "acc": (tp + tn) / n if n else 0.0}


def pass1_acc(groups):
    n = hit = 0
    for g in groups:
        for r in g:
            n += 1
            hit += int(expected(r) == predicted(r))
    return hit / n if n else 0.0


def passk_acc(groups):
    n = hit = 0
    for g in groups:
        if not g:
            continue
        n += 1
        hit += int(expected(g[0]) in {predicted(r) for r in g})
    return hit / n if n else 0.0


# ── reporting ─────────────────────────────────────────────────────────────────
def table(rows, headers):
    cells = [[f"{c:.4f}" if isinstance(c, float) else str(c) for c in r] for r in rows]
    w = [max(len(x) for x in col) for col in zip(headers, *cells)] if cells else [len(h) for h in headers]
    fmt = lambda r: " | ".join(c.ljust(wi) for c, wi in zip(r, w))
    return "\n".join([fmt(headers), "-+-".join("-" * wi for wi in w), *map(fmt, cells)])


def latest_results_path(results_root: Path, run_name: str) -> Path:
    eval_root = results_root / run_name / "vf" / "evals" / f"med_app_env--{run_name}"
    candidates = list(eval_root.rglob("results.jsonl"))
    if not candidates:
        raise FileNotFoundError(f"no results.jsonl under {eval_root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    models = {}
    for run_name in args.runs:
        try:
            models[run_name] = latest_results_path(args.results_root, run_name)
        except FileNotFoundError as error:
            print(f"[warn] {error}")

    reports: dict[str, dict] = {}
    for name, path in models.items():
        p = Path(path)
        if not p.exists():
            print(f"[warn] {name}: {path} not found")
            continue
        records = load(p)
        groups = group_rollouts(records, args.k)
        labels = {expected(r) for g in groups for r in g} | {
            prediction
            for g in groups
            for r in g
            if (prediction := predicted(r)) != ERROR_LABEL
        }
        report = {
            "n_samples": len(groups),
            "n_rollouts": sum(len(g) for g in groups),
            "pass@1": {"acc": pass1_acc(groups), "trigger": pass1_trigger(groups), "per_class": pass1_per_class(groups, labels)},
        }
        if args.k != 1:
            report[f"pass@{args.k}"] = {
                "acc": passk_acc(groups),
                "trigger": passk_trigger(groups),
                "per_class": passk_per_class(groups, labels),
            }
        reports[name] = report
    if not reports:
        return
    ms = list(reports)

    print("\n=== Dataset sizes ===")
    print(
        table(
            [["n_samples"] + [reports[m]["n_samples"] for m in ms], ["n_rollouts"] + [reports[m]["n_rollouts"] for m in ms]],
            ["metric"] + ms,
        )
    )

    variants = ["pass@1"] if args.k == 1 else ["pass@1", f"pass@{args.k}"]
    for variant in variants:
        print(f"\n======== {variant} ========")
        print("\n=== Overall accuracy ===")
        print(table([["acc"] + [reports[m][variant]["acc"] for m in ms]], ["metric"] + ms))
        print("\n=== Trigger (tool vs. no_tool) ===")
        print(table([[k] + [reports[m][variant]["trigger"][k] for m in ms] for k in ("P", "R", "F1", "acc")], ["metric"] + ms))
        print("\n=== Per-tool P / R / F1 ===")
        all_labels = sorted({l for m in ms for l in reports[m][variant]["per_class"]})
        rows = [[l] + [v for m in ms for v in reports[m][variant]["per_class"].get(l, (0.0, 0.0, 0.0, 0))[:3]] for l in all_labels]
        print(table(rows, ["label"] + [f"{m}_{k}" for m in ms for k in ("P", "R", "F1")]))

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "k": args.k,
            "models": {
                name: {"input": str(models[name]), **report}
                for name, report in reports.items()
            },
        }
        args.output_json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"\nWrote {args.output_json}")


if __name__ == "__main__":
    main()
