import sys
from pathlib import Path

# Put the project root on sys.path so `from src....` works when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import truststore
from tqdm.auto import tqdm
from practise.m01_agentic_rag.ingest import load_faq_data

# Use the OS trust store (macOS keychain) so the corporate Zscaler root CA
# is trusted, just like it is in the browser. Must run before any requests.
truststore.inject_into_ssl()

# one time - load model
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

# ingestion stage - convert documents to vectors - in batches
documents = load_faq_data()
texts = []
for doc in documents:
    text = doc["question"] + " " + doc["answer"]
    texts.append(text)

BATCH_SIZE = 50
vectors = []
print("Creating Embeddings: ")
for i in tqdm(range(0, len(texts), BATCH_SIZE)):
    batch = texts[i : i + BATCH_SIZE]
    vectors.extend(model.encode(batch))

vectors_arr = np.array(vectors)

# user query - retrieval
query = "Can I still join the course after the start date?"
v_query = model.encode(query)
scores = vectors_arr.dot(
    v_query
)  # dot product because model already outputs cosine values

# top document - max value
idx = np.argmax(scores)  # argmax - return largest value index
print(idx)
print(documents[idx])

# top-k documents
# argsort sorts from lowest to highest, so last element in highest
top5 = np.argsort(scores)[-5:]
top5 = top5[::-1]

# OR just negate the scores
top5 = np.argsort(-scores)[:5]  # top 5
