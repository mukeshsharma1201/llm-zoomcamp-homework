import sys
from pathlib import Path

# Put the project root on sys.path so `from src....` works when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import truststore
from sentence_transformers import SentenceTransformer
from sqlitesearch import VectorSearchIndex
from tqdm.auto import tqdm

from practise.m01_agentic_rag.ingest import load_faq_data

# Use the OS trust store (macOS keychain) so the corporate Zscaler root CA
# is trusted, just like it is in the browser. Must run before any requests.
truststore.inject_into_ssl()

################# INGESTION PIPELINE #################
# this will actually persist the document embeddings so that it can be loaded
# by any other process and used for retrieval

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

# create persistent index in sqlite db
vs_index = VectorSearchIndex(
    keyword_fields=["course"], mode="ivf", db_path="faq_vectors2.db"
)
vs_index.fit(vectors, documents)

# sqlitesearch supports three ANN modes:

# lsh (default): up to 100K vectors, random hyperplane projections
# ivf: 10K-500K vectors, K-means clustering
# hnsw: 10K-1M+ vectors, proximity graph (highest recall)


query = "I just discovered the course. Can I still join it?"
query_vector = model.encode(query)

results = vs_index.search(query_vector, num_results=5)
print(results)

vs_index.close()
