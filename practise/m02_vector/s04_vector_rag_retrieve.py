import sys
from pathlib import Path
from textwrap import dedent

# Put the project root on sys.path so `from src....` works when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


from sentence_transformers import SentenceTransformer
from openai import OpenAI
from practise.m01_agentic_rag.rag_helper import RAGBase
from practise.config import get_llm_provider_config
from sqlitesearch import VectorSearchIndex

################# RETRIEVAL #################
model = SentenceTransformer("all-MiniLM-L6-v2")
llm_cfg = get_llm_provider_config()
client = OpenAI(base_url=llm_cfg.base_url, api_key=llm_cfg.api_key)

custom_system_instruction = dedent("""
            You're a course teaching assistant.
            Answer the QUESTION based on the CONTEXT from the FAQ database.
            Use only the facts from the CONTEXT when answering the QUESTION.
""")


query = "I just discovered the course. Can I still join it?"
query_vector = model.encode(query)

# re-open the connection to previously created index
vs_index = VectorSearchIndex(
    keyword_fields=["course"], mode="ivf", db_path="faq_vectors2.db"
)


################# PREPARE RAG BASE RUNNER for Vector search #################
class RAGVector(RAGBase):
    def __init__(self, *args, embedder, **kwargs):
        super().__init__(*args, **kwargs)
        if not embedder:
            raise ValueError("embedder is required")
        self.embedder = embedder

    def search(self, query, num_results=5):
        """Override the search method so that it can support vector search,
        instead of a text search.
        """
        query_vector = self.embedder.encode(query)
        filter_dict = {"course": self.course}
        return self.index.search(
            query_vector, num_results=num_results, filter_dict=filter_dict
        )


assistant = RAGVector(
    embedder=model,
    index=vs_index,
    llm_client=client,
    model=llm_cfg.model,
    instructions=custom_system_instruction,
)

print(f"{'*' * 30} RAG {'*' * 30}")
user_query = "its late, can I still sign up?"
print("Query: ", user_query)
answer = assistant.rag(user_query)
print(answer)

# close connection to the DB
vs_index.close()
