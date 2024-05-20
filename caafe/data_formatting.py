import json
from client import chat_complete
import re
from change_utility import (update_long, extract_parts)
# 等跑完了改成good
log_path = "/home/jiahe/ML/Self_instruct_CAAFE/caafe/log/good.jsonl"
# 等跑完了改成目标目录
output_path = "/home/jiahe/ML/Self_instruct_CAAFE/LLaMA-Factory/data/caafe_finetuning_data.json"
change_prompt = {}
output_data = []

# 从老描述生成简化描述


def get_new_description(old_description):
    if old_description == None or old_description == "":
        return ""
    if old_description in change_prompt:
        return change_prompt[old_description]
    else:
        if "vowel" in old_description:
            print("Found")
            new_d = '''A speech dataset containing 11 steady state vowels of British English, with LPC-derived log area ratios, suitable for speaker-independent recognition."vowel.data". This consists of a three dimensional array: vowel data [speaker, vowel, input]. The speakers are indexed by integers 0-89.  (Actually, there are fifteen individual speakers, each saying each vowel six times.)  The vowels are indexed by integers 0-10.  For each utterance, there are ten floating-point input values, with array indices 0-9.The problem is to train the network as well as possible using only on data from "speakers" 0-47, and then to test the network on speakers 48-89,reporting the number of correct classifications in the test set.'''
            change_prompt[old_description] = new_d
            return new_d
        print("Asked!")
        simlify_messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            {
                "role": "user",
                "content": "You will simplify the data description,only preserve those which is critical for feature engineering. You should only response the simplified description .Your answer should begin with \"```begin\" .Dataset description:\n"+old_description,
            },
        ]
        _, simplified_description = chat_complete(
            messages=simlify_messages,
            stop=["```end"],
            temperature=0.5,
            max_tokens=500,
        )
        simplified_description = simplified_description.replace("```begin", "").replace(
            "```", "").replace("<end>", "")

        change_prompt[old_description] = simplified_description
        return simplified_description


def get_new_samples(old_samples):
    new_samples = re.sub(r"NaN-freq \[.*\], ", "", old_samples)
    new_samples = re.sub(r",.*]", "]", new_samples)
    new_samples = new_samples.replace("Samples", "Sample")
    return new_samples


def get_new_prompt(old_prompt):
    update_long(old_prompt)
    old_description, old_samples = extract_parts(old_prompt)
    old_prompt = old_prompt.replace(old_samples, get_new_samples(old_samples))
    if old_description != None:
        new_description = get_new_description(old_description)
        old_prompt = old_prompt.replace(old_description, new_description)
    return old_prompt


with open(log_path, "r") as f:
    lines = f.readlines()
for line in lines:
    original_data = json.loads(line)
    old_prompt = original_data["messages"][1]["content"]
    original_data["messages"][1]["content"] = get_new_prompt(old_prompt)
    formatted_data = {
        "messages": original_data["messages"]
    }
    formatted_data["messages"].append({
        "role": "assistant",
        "content": original_data["response"]
    })
    if float(original_data["improvement_roc"]) > 0 and float(original_data["improvement_acc"]) > 0:
        output_data.append(formatted_data)
with open(output_path, "w") as f:
    f.write(json.dumps(output_data, indent=2))
