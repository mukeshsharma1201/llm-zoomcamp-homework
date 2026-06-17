from openai import OpenAI

from config import get_llm_provider_config

cfg = get_llm_provider_config()
client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)

response = client.chat.completions.create(
    model=cfg.model,
    messages=[
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "Write a python function to sort a list."},
    ],
)

print(type(response))

print(response)

print(response.choices[0].message.content)
