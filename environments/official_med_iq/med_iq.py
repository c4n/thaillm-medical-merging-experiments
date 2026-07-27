"""Published med-IQ evaluator with an offline/deferred judge option.

The scoring logic is copied from vistec-AI/thaillm-medical-post-training at
commit 73772633663dfe02eff558a85eacbac9f617d329.  ``deferred`` mode changes
only when the OpenRouter response judge runs; JSON parsing and the two local
reward functions intentionally retain the strict published behavior.
"""

import json
import os
import re

from datasets import disable_caching, load_dataset
from openai import AsyncOpenAI
import verifiers as vf

disable_caching()

UPSTREAM_COMMIT = "73772633663dfe02eff558a85eacbac9f617d329"
JUDGE_MODEL = "deepseek/deepseek-v4-flash"
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


def extract_question(prompt_str):
    marker = '{"answer": "unknown", "citations": []}'
    after_marker = prompt_str.split(marker, 1)[1]
    question = after_marker.split("Facts:", 1)[0]
    return question.strip()


def parse_json(response):
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return None


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


def load_environment(**kwargs):
    print(
        "Loading official med_iq evaluator from {} at upstream {}".format(
            __file__,
            UPSTREAM_COMMIT,
        ),
        flush=True,
    )
    dataset = load_dataset("ThaiLLM/med-iq")
    dataset = dataset.map(preprocess, load_from_cache_file=False)

    async def format_reward(completion, info):
        response = parse_json(completion[-1]["content"])
        if not isinstance(response, dict):
            return 0.0
        if not isinstance(response.get("answer"), str):
            return 0.0
        citations = response.get("citations")
        if not isinstance(citations, list) or not all(
            isinstance(citation, str) for citation in citations
        ):
            return 0.0
        return 1.0

    async def citations_reward(completion, info):
        response = parse_json(completion[-1]["content"])
        if not isinstance(response, dict):
            return 0.0
        predicted = response.get("citations")
        if not isinstance(predicted, list) or not predicted:
            return 0.0
        if not all(citation in info["citation_ids"] for citation in predicted):
            return 0.0
        if not any(citation in info["citations"] for citation in predicted):
            return 0.0
        return (
            sum(1 for citation in predicted if citation in info["citations"])
            / len(predicted)
        )

    judge_mode = os.getenv("MED_IQ_JUDGE_MODE", "online")
    if judge_mode == "online":
        judge_client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
        judge_rubric = vf.JudgeRubric(
            judge_client=judge_client,
            judge_model=JUDGE_MODEL,
        )

        async def response_reward(
            completion,
            info,
            state,
            judge_client,
            judge_model,
            **kwargs,
        ):
            predicted = parse_json(completion[-1]["content"])
            if not isinstance(predicted, dict) or not isinstance(
                predicted.get("answer"), str
            ):
                return 0.0
            judge_prompt = JUDGE_TEMPLATE.format(
                question=info["question"],
                answer=info["answer"],
                response=predicted["answer"],
            )
            response = await judge_client.chat.completions.create(
                model=judge_model,
                messages=[{"role": "user", "content": judge_prompt}],
            )
            verdict = response.choices[0].message.content or ""
            state["judge_response"] = verdict
            return 1.0 if verdict.strip().upper().startswith("A") else 0.0

        judge_rubric.add_reward_func(response_reward, weight=1.0)
    elif judge_mode == "deferred":

        async def response_reward(completion, info, state):
            predicted = parse_json(completion[-1]["content"])
            if not isinstance(predicted, dict) or not isinstance(
                predicted.get("answer"), str
            ):
                return 0.0
            state["judge_response"] = "deferred"
            return 0.0

        judge_rubric = vf.Rubric(funcs=[response_reward], weights=[1.0])
    else:
        raise ValueError("MED_IQ_JUDGE_MODE must be 'online' or 'deferred'")

    verifiable_rubric = vf.Rubric(
        funcs=[format_reward, citations_reward],
        weights=[0.1, 1.0],
    )
    rubric = vf.RubricGroup([verifiable_rubric, judge_rubric])
    return vf.SingleTurnEnv(
        dataset=dataset["train"],
        eval_dataset=dataset["test"],
        rubric=rubric,
    )
