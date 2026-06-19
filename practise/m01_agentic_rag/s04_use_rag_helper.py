import sys
from pathlib import Path
from textwrap import dedent

# Put the project root on sys.path so `from src....` works when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ingest import build_text_index, load_faq_data
from openai import OpenAI
from rag_helper import RAGBase

from practise.config import get_llm_provider_config

docs = load_faq_data(course="llm-zoomcamp")
index = build_text_index(docs, clear=True)
llm_cfg = get_llm_provider_config()
client = OpenAI(api_key=llm_cfg.api_key, base_url=llm_cfg.base_url)

custom_system_instruction = dedent("""
                                You're a course teaching assistant.
                                Answer the QUESTION based on the CONTEXT from the FAQ database.
                                Use only the facts from the CONTEXT when answering the QUESTION.
                            """)

assistant = RAGBase(
    index=index,
    llm_client=client,
    model=llm_cfg.model,
    instructions=custom_system_instruction,
)

answer = assistant.rag("How do I run lamma?")

print(answer)
