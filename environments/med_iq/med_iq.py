import os
import json
import re

from openai import AsyncOpenAI
import verifiers as vf
from datasets import load_dataset, disable_caching

disable_caching()

JUDGE_TEMPLATE = """\
Your job is to evaluate whether a predicted answer matches the reference answer for a medical question in Thai.

Question: {question}
Reference Answer: {answer}
Predicted Answer: {response}

Instructions:
- The predicted answer is CORRECT if it conveys the same key medical information as the reference (same condition, same recommended action).
- Paraphrasing, translation, and minor wording differences are acceptable.
- If the reference is "unknown", the predicted answer is CORRECT only if it also abstains.
- Ignore stylistic differences; focus on clinical content.

Grade the predicted answer as one of:
A: CORRECT
B: INCORRECT

Just return the letter "A" or "B", with no text around it.
""".strip()


def extract_question(prompt_str: str) -> str:
    marker = '{"answer": "unknown", "citations": []}'
    after_marker = prompt_str.split(marker, 1)[1]
    question = after_marker.split("Facts:", 1)[0]
    return question.strip()


def parse_json(response):
    try:
        return json.loads(response)
    except (json.JSONDecodeError, TypeError):
        return None


def parse_json_relaxed(response):
    """Recover content for semantic scoring while keeping format scoring strict."""
    if not isinstance(response, str):
        return None

    text = response.strip()
    text = re.sub(r"^assistant\s*:?[\s\r\n]*", "", text, count=1, flags=re.IGNORECASE)
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, count=1, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text, count=1)

    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        return None
    parsed = parse_json(text[start : end + 1])
    if not isinstance(parsed, dict):
        return None

    citations = parsed.get("citations")
    if isinstance(citations, str):
        # The published SFT-IQ checkpoint emits JSON such as
        # {"citations": "[1]"}. Normalize only this narrow bracketed-ID form.
        if not re.fullmatch(r"\s*\[(?:\s*[\w-]+\s*(?:,\s*[\w-]+\s*)*)?\]\s*", citations):
            return None
        parsed = dict(parsed)
        parsed["citations"] = re.findall(r"[\w-]+", citations)
    return parsed


def preprocess(example):
    facts_section = example["prompt"][0]["content"].split("Facts:", 1)[1]
    answer = parse_json(example["completion"][0]["content"])
    return {
        "info": {
            "citation_ids": re.findall(r"\[(\w+)\]", facts_section),
            "citations": answer.get("citations"),
            "answer": answer.get("answer"),
            "question": extract_question(example["prompt"][0]["content"]),
        }
    }


def load_environment(**kwargs) -> vf.Environment:
    """
    Loads a custom environment.
    """
    print(f"Loading med_iq evaluator from {__file__}", flush=True)
    dataset = load_dataset("ThaiLLM/med-iq")
    dataset = dataset.map(preprocess, load_from_cache_file=False)
    debug_printed = False
    judge_debug_printed = False

    async def format_reward(completion, info):
        response = parse_json(completion[-1]["content"])
        if not isinstance(response, dict):
            return 0.0
        if not isinstance(response.get("answer"), str):
            return 0.0
        citations = response.get("citations")
        if not isinstance(citations, list) or not all(isinstance(c, str) for c in citations):
            return 0.0
        return 1.0

    async def citations_reward(completion, info):
        nonlocal debug_printed
        raw_response = completion[-1]["content"]
        response = parse_json_relaxed(raw_response)
        if os.getenv("MED_IQ_DEBUG") == "1" and not debug_printed:
            print(f"MED_IQ_RAW_COMPLETION={raw_response!r}", flush=True)
            print(f"MED_IQ_RELAXED_PARSE={response!r}", flush=True)
            debug_printed = True
        if not isinstance(response, dict):
            return 0.0
        pred = response.get("citations")
        if not isinstance(pred, list) or not pred:
            return 0.0
        if not all(c in info["citation_ids"] for c in pred):
            return 0.0
        if not any(c in info["citations"] for c in pred):
            return 0.0
        return sum(1 for c in pred if c in info["citations"]) / len(pred)

    judge_mode = os.getenv("MED_IQ_JUDGE_MODE", "online")
    judge_client = None
    if judge_mode == "online":
        judge_client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
    judge_model = "deepseek/deepseek-v4-flash"

    async def response_reward(completion, info, state):
        nonlocal judge_debug_printed
        pred = parse_json_relaxed(completion[-1]["content"])
        if not isinstance(pred, dict) or not isinstance(pred.get("answer"), str):
            return 0.0
        if judge_mode == "deferred":
            state["judge_response"] = "deferred"
            return 0.0
        if judge_client is None:
            raise RuntimeError("OpenRouter judge client is unavailable in online mode")
        judge_prompt = JUDGE_TEMPLATE.format(question=info["question"], answer=info["answer"], response=pred["answer"])
        try:
            resp = await judge_client.chat.completions.create(
                model=judge_model,
                messages=[{"role": "user", "content": judge_prompt}],
            )
        except Exception as exc:
            if os.getenv("MED_IQ_DEBUG") == "1":
                print(f"MED_IQ_JUDGE_ERROR={type(exc).__name__}: {exc}", flush=True)
            raise
        verdict = resp.choices[0].message.content or ""
        if os.getenv("MED_IQ_DEBUG") == "1" and not judge_debug_printed:
            print(f"MED_IQ_JUDGE_QUESTION={info['question']!r}", flush=True)
            print(f"MED_IQ_REFERENCE_ANSWER={info['answer']!r}", flush=True)
            print(f"MED_IQ_PREDICTED_ANSWER={pred['answer']!r}", flush=True)
            print(f"MED_IQ_JUDGE_VERDICT={verdict!r}", flush=True)
            judge_debug_printed = True
        state["judge_response"] = verdict
        return 1.0 if verdict.strip().upper().startswith("A") else 0.0

    judge_rubric = vf.Rubric(funcs=[response_reward], weights=[1.0])
    verifiable_rubric = vf.Rubric(funcs=[format_reward, citations_reward], weights=[0.1, 1.0])
    rubric = vf.RubricGroup([verifiable_rubric, judge_rubric])
    return vf.SingleTurnEnv(dataset=dataset["train"], eval_dataset=dataset["test"], rubric=rubric)
