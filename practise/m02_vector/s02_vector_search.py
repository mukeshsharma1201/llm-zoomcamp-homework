import sys
from pathlib import Path
from textwrap import dedent

# Put the project root on sys.path so `from src....` works when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import truststore
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
from minsearch import VectorSearch
from openai import OpenAI
from practise.m01_agentic_rag.ingest import load_faq_data
from practise.m01_agentic_rag.rag_helper import RAGBase
from practise.config import get_llm_provider_config

# Use the OS trust store (macOS keychain) so the corporate Zscaler root CA
# is trusted, just like it is in the browser. Must run before any requests.
truststore.inject_into_ssl()

################# INGESTION #################
documents = load_faq_data()
model = SentenceTransformer("all-MiniLM-L6-v2")
BATCH_SIZE = 50
vectors = []
print(f"creating embeddings for {len(documents)} docs")
texts = [doc["question"] + " " + doc["answer"] for doc in documents]
for i in tqdm(range(0, len(documents), BATCH_SIZE)):
    batch = texts[i : i + BATCH_SIZE]
    vectors.extend(model.encode(batch))

# create a matrix that contains all individual vectors
matX = np.array(vectors)

# create an index that is filterable
vindex = VectorSearch(keyword_fields=["course"])
vindex.fit(matX, documents)


################# RETRIEVAL #################
query = "I just discovered the course. Can I still join it?"
query_vector = model.encode(query)

results = vindex.search(
    query_vector, filter_dict={"course": "llm-zoomcamp"}, num_results=5
)

print(results)


#############################################################################
#############################################################################


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


################# RAG #################
llm_cfg = get_llm_provider_config()
client = OpenAI(base_url=llm_cfg.base_url, api_key=llm_cfg.api_key)

custom_system_instruction = dedent("""
            You're a course teaching assistant.
            Answer the QUESTION based on the CONTEXT from the FAQ database.
            Use only the facts from the CONTEXT when answering the QUESTION.
""")


assistant = RAGVector(
    embedder=model,
    index=vindex,
    llm_client=client,
    model=llm_cfg.model,
    instructions=custom_system_instruction,
)
print(f"{'*' * 30} RAG {'*' * 30}")
user_query = "I just found out about the program, can I still sign up?"
print("Query: ", user_query)
answer = assistant.rag(user_query)
print(answer)
