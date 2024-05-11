from openai import OpenAI
import json

# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = models.data[0].id

log_path = "/home/jiahe/ML/Self_instruct_CAAFE/caafe/log/all.jsonl"


def chat_complete(messages, stop=["```end"], temperature=0.5, max_tokens=256):
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        stop=stop,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    response = completion.choices[0].message.content

    log_entry = {
        "messages": messages,
        "response": response
    }

    try:
        with open(log_path, 'a') as log_file:
            json.dump(log_entry, log_file)
            log_file.write("\n")  # Add newline to separate entries
    except Exception as e:
        print(f"Warning: Could not write to log file: {e}")

    return completion, response
