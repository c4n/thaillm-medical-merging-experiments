import json
import random

from datasets import DatasetDict, concatenate_datasets, load_dataset

HEALTHCARE_TOOLS = (
    "search_medical_facts",
    "prescreen",
    "get_health_emergency_contact",
    "create_appointment",
    "create_reminder",
    "list_appointment",
    "list_reminder",
)


def extract_thaillm_negatives(example):
    conversation = json.loads(example["conversations"])
    system_prompt = next((t["content"] for t in conversation if t["role"] == "system"), "")
    is_thaillm = "ThaiLLM" in system_prompt or "Medical AI Assistant" in system_prompt
    is_negative = example["tool_name"] == "negatives"
    return is_thaillm and is_negative

def extract_thaillm_positives(example):
    conversation = json.loads(example["conversations"])
    system_prompt = next((t["content"] for t in conversation if t["role"] == "system"), "")
    is_thaillm = "ThaiLLM" in system_prompt or "Medical AI Assistant" in system_prompt
    is_positive = example["tool_name"] in HEALTHCARE_TOOLS
    return is_thaillm and is_positive

def extract_tulu_negatives(example):
    conversation = json.loads(example["conversations"])
    system_prompt = next((t["content"] for t in conversation if t["role"] == "system"), "")
    is_tulu = "ThaiLLM" not in system_prompt and "Medical AI Assistant" not in system_prompt
    is_negative = example["tool_name"] == "negatives"
    return is_tulu and is_negative

dataset = load_dataset("/scratch_aisg/patomp/prime-rl/datasets/med-app-instruct", split="qwen3_235b_a22_thinking_2507")
thaillm_negative = dataset.filter(extract_thaillm_negatives) # 21834 samples
thaillm_positive = dataset.filter(extract_thaillm_positives) # 19405 samples
tulu_negative = dataset.filter(extract_tulu_negatives) # 335238 samples
tulu_negative = tulu_negative.shuffle(seed=42).select(range(10_000))

# Split train test
test_idx, train_idx = [], []
for tool in HEALTHCARE_TOOLS:
    indices = [i for i, t in enumerate(thaillm_positive["tool_name"]) if t == tool]
    random.shuffle(indices)
    n_test = max(1, int(len(indices) * 0.1))
    test_idx.extend(indices[:n_test])
    train_idx.extend(indices[n_test:])

thaillm_positive = DatasetDict({
    "train": thaillm_positive.select(train_idx),
    "test": thaillm_positive.select(test_idx)
})

# for tool in HEALTHCARE_TOOLS:
#     count = thaillm_positive["test"]["tool_name"].count(tool)
#     print(f"{tool}: {count} ({count/len(thaillm_positive["test"])}) test samples")

# for tool in HEALTHCARE_TOOLS:
#     count = thaillm_positive["train"]["tool_name"].count(tool)
#     print(f"{tool}: {count} ({count/len(thaillm_positive["train"])}) train samples")

thaillm_negative = thaillm_negative.train_test_split(test_size=0.1)
tulu_negative = tulu_negative.train_test_split(test_size=0.1)

# Concatenate
# dataset = DatasetDict({
#     "train": concatenate_datasets([thaillm_positive["train"], thaillm_negative["train"], tulu_negative["train"]]),
#     "test": concatenate_datasets([thaillm_positive["test"], thaillm_negative["test"], tulu_negative["test"]])
# })

dataset = DatasetDict({
    "train": concatenate_datasets([thaillm_positive["train"], thaillm_negative["train"]]),
    "test": concatenate_datasets([thaillm_positive["test"], thaillm_negative["test"]])
})

def format(example):
    conversation = json.loads(example["conversations"])
    prompt = [turn for turn in conversation if turn["role"] in ("system", "user")]
    return {"prompt": prompt, "answer": example["tool_name"]}
dataset = dataset.map(format)
dataset.save_to_disk("/scratch_aisg/patomp/prime-rl/datasets/med-app-env-no-tulu")