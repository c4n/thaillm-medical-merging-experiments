#!/usr/bin/env python3
"""Post-hoc OpenRouter judging for med-IQ results generated on offline GPU nodes."""

import argparse
import asyncio
import json
import os
import re
from pathlib import Path

from openai import AsyncOpenAI

from med_iq import JUDGE_TEMPLATE, parse_json, parse_json_relaxed


def strict_format_score(raw: str) -> float:
    parsed = parse_json(raw)
    if not isinstance(parsed, dict) or not isinstance(parsed.get("answer"), str):
        return 0.0
    citations = parsed.get("citations")
    return float(
        isinstance(citations, list)
        and all(isinstance(citation, str) for citation in citations)
    )


def citations_score(parsed: dict | None, info: dict) -> float:
    if not isinstance(parsed, dict):
        return 0.0
    predicted = parsed.get("citations")
    if not isinstance(predicted, list) or not predicted:
        return 0.0
    if not all(isinstance(citation, str) for citation in predicted):
        return 0.0
    citation_ids = info.get("citation_ids") or []
    references = info.get("citations") or []
    if not all(citation in citation_ids for citation in predicted):
        return 0.0
    if not any(citation in references for citation in predicted):
        return 0.0
    return sum(citation in references for citation in predicted) / len(predicted)


def parse_verdict(verdict: str) -> str | None:
    match = re.fullmatch(r"\s*`{0,3}\s*([ABab])\s*[.]?\s*`{0,3}\s*", verdict)
    return match.group(1).upper() if match else None


def raw_completion(record: dict) -> str:
    completion = record.get("completion") or []
    if not completion or not isinstance(completion[-1], dict):
        return ""
    content = completion[-1].get("content")
    return content if isinstance(content, str) else ""


async def main_async(args: argparse.Namespace) -> None:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is not set")

    records = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    latest: dict[int, dict] = {}
    if args.output.exists():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                latest[item["source_index"]] = item

    completed = {
        index
        for index, item in latest.items()
        if not item.get("judge_error")
    }
    pending = [(index, record) for index, record in enumerate(records) if index not in completed]
    print(f"Input rows: {len(records)}; already scored: {len(completed)}; pending: {len(pending)}")

    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    semaphore = asyncio.Semaphore(args.concurrency)

    async def score(index: int, record: dict) -> dict:
        raw = raw_completion(record)
        parsed = parse_json_relaxed(raw)
        info = record.get("info") if isinstance(record.get("info"), dict) else {}
        result = {
            "source_index": index,
            "example_id": record.get("example_id"),
            "format_reward": strict_format_score(raw),
            "citations_reward": citations_score(parsed, info),
            "response_reward": 0.0,
            "predicted_answer": parsed.get("answer") if isinstance(parsed, dict) else None,
            "reference_answer": info.get("answer"),
            "judge_response": None,
            "judge_error": None,
        }
        if not isinstance(result["predicted_answer"], str):
            result["combined_reward"] = result["format_reward"] * 0.1 + result["citations_reward"]
            return result

        prompt = JUDGE_TEMPLATE.format(
            question=info.get("question", ""),
            answer=info.get("answer", ""),
            response=result["predicted_answer"],
        )
        try:
            async with semaphore:
                response = await client.chat.completions.create(
                    model=args.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                )
            verdict_text = response.choices[0].message.content or ""
            verdict = parse_verdict(verdict_text)
            result["judge_response"] = verdict_text
            if verdict is None:
                result["judge_error"] = "Judge did not return exactly A or B"
            else:
                result["response_reward"] = float(verdict == "A")
        except Exception as exc:
            result["judge_error"] = f"{type(exc).__name__}: {exc}"

        result["combined_reward"] = (
            result["format_reward"] * 0.1
            + result["citations_reward"]
            + result["response_reward"]
        )
        return result

    tasks = [asyncio.create_task(score(index, record)) for index, record in pending]
    with args.output.open("a", encoding="utf-8") as output_file:
        for count, task in enumerate(asyncio.as_completed(tasks), start=1):
            result = await task
            latest[result["source_index"]] = result
            output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            output_file.flush()
            if count % 20 == 0 or count == len(tasks):
                print(f"Scored {count}/{len(tasks)} pending rows", flush=True)

    ordered = [latest[index] for index in sorted(latest)]
    n = len(ordered)
    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "judge_model": args.model,
        "n": n,
        "format_reward": sum(item["format_reward"] for item in ordered) / n if n else 0.0,
        "citations_reward": sum(item["citations_reward"] for item in ordered) / n if n else 0.0,
        "response_reward": sum(item["response_reward"] for item in ordered) / n if n else 0.0,
        "combined_reward": sum(item["combined_reward"] for item in ordered) / n if n else 0.0,
        "judge_errors": sum(bool(item.get("judge_error")) for item in ordered),
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    os.umask(0o002)
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="vf-eval results.jsonl")
    parser.add_argument("--output", type=Path, required=True, help="Incremental scored JSONL")
    parser.add_argument("--model", default="deepseek/deepseek-v4-flash")
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
