import sys
from pathlib import Path

# Put the project root on sys.path so `from src....` works when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from openai import OpenAI  # noqa: E402

from practise.config import get_llm_provider_config  # noqa: E402

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
