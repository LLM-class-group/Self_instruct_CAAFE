from openai import OpenAI

# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
llama3 = models.data[0].id

prompt = input("prompt:")

completion = client.chat.completions.create(
    model=llama3,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ],
    stop=["```end"],
    temperature=0.5,
    max_tokens=500,
)

print("Chat response:", completion.choices[0].message.content)
