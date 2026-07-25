import json
import re
import verifiers as vf
from datasets import load_dataset

TOOL_PATTERN = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def extract_tool_call(text: str) -> dict | None:
    if not text:
        return None
    match = TOOL_PATTERN.search(text)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return None


def load_environment(**kwargs) -> vf.Environment:
    dataset = load_dataset("ThaiLLM/med-tool-use")

    async def correct_tool_reward(completion, answer) -> float:
        response = completion[-1]["content"]
        has_tag = "<tool_call>" in response
        tool_call = extract_tool_call(response)
        if answer == "negatives":
            if has_tag:
                return -1.0
            return 1.0 if 30 <= len(response) <= 3000 else 0.5
        if tool_call is None:
            return -0.5 if has_tag else 0.0
        return 1.0 if tool_call.get("name") == answer else -0.5 

    rubric = vf.Rubric(funcs=[correct_tool_reward], weights=[1.0])
    return vf.SingleTurnEnv(dataset=dataset["train"], rubric=rubric, eval_dataset=dataset["test"])
