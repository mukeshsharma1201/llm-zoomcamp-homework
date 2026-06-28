import sys
from pathlib import Path
from tqdm.auto import tqdm
import numpy as np

# Put the project root on sys.path so `from src....` works when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import truststore
from practise.m02_vector.onnx.embedder import Embedder
from practise.m01_agentic_rag.ingest import load_faq_data

# Use the OS trust store (macOS keychain) so the corporate Zscaler root CA
# is trusted, just like it is in the browser. Must run before any requests.
truststore.inject_into_ssl()
embed = Embedder()

documents = load_faq_data()
BATCH_SIZE = 50
vectors = []
print(f"creating embeddings for {len(documents)} docs")
texts = [doc["question"] + " " + doc["answer"] for doc in documents]

for i in tqdm(range(0, len(texts), BATCH_SIZE)):
    batch = texts[i : i + BATCH_SIZE]
    vectors.extend(embed.encode_batch(batch))

X = np.array(vectors)


### search

query = "Can I still join the course after the start date?"
v_query = embed.encode(query)

scores = X.dot(v_query)
idx = np.argmax(scores)

print(documents[idx])
