import sys
from pathlib import Path

# Put the project root on sys.path so `from src....` works when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ingest import build_text_index, load_faq_data
from openai import OpenAI
from rag_helper import RAGBase

from practise.config import get_llm_provider_config

docs = load_faq_data(course="llm-zoomcamp")
index = build_text_index(docs)
llm_cfg = get_llm_provider_config()
client = OpenAI(api_key=llm_cfg.api_key, base_url=llm_cfg.base_url)

assistant = RAGBase(index=index, llm_client=client, model=llm_cfg.model)

answer = assistant.rag("I just discovered the course. Can I join now?")

print(answer)
