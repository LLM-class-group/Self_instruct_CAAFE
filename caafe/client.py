from datetime import datetime
from openai import OpenAI, OpenAIError

# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = models.data[0].id


def chat_complete(messages, stop=["```end"], temperature=0.5, max_tokens=256):
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            stop=stop,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    except OpenAIError as e:
        print(f"LLM request failed with: {e}")
        # kill program
        exit(1)

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        stop=stop,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    response = completion.choices[0].message.content

    # log_entry = {
    #     "messages": messages,
    #     "response": response
    # }

    # try:
    #     with open(log_path, 'a') as log_file:
    #         json.dump(log_entry, log_file)
    #         log_file.write("\n")
    # except Exception as e:
    #     print(f"Warning: Could not write to log file: {e}")

    return completion, response


def get_time():
    current_time = datetime.now()
    standardized_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    # return in str
    return standardized_time


log_file_path = 'caafe/log/important.txt'
with open(log_file_path, 'w') as log_file:
    log_file.write('')


def print_important(*args, **kwargs):
    # 格式化输出内容
    message = ' '.join(map(str, args))

    # 输出到控制台
    print(message, **kwargs)

    # 输出到日志文件
    with open(log_file_path, 'a') as log_file:
        log_file.write(message + '\n')
