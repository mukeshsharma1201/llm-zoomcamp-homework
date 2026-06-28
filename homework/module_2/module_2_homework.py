import sys
from pathlib import Path

# Put the project root on sys.path so `from homework....` works when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from minsearch import Index, VectorSearch
from tqdm.auto import tqdm

from homework.module_1.ingest import chunk_documents, fetch_documents
from homework.module_2.onnx.embedder import Embedder

############################## Q1 ##############################
documents = fetch_documents()
embed = Embedder()
user_query = "How does approximate nearest neighbor search work?"
uq_vector = embed.encode(user_query)
print("Q1: ", uq_vector[0])  # -0.02


############################## Q2 ##############################
target_file = "02-vector-search/lessons/07-sqlitesearch-vector.md"
file = [f["content"] for f in documents if f["filename"] == target_file]
file_vector = embed.encode(file[0])
cosine_similarity = uq_vector.dot(file_vector)
print("Q2: ", cosine_similarity)  # 0.361070280302606


############################## Q3 ##############################
chunks = chunk_documents(documents)
vectors = []
BATCH_SIZE = 50
chunks_content = [c["content"] for c in chunks]

for i in tqdm(range(0, len(chunks_content), BATCH_SIZE)):
    batch_embedding = embed.encode_batch(chunks_content[i : i + BATCH_SIZE])
    vectors.extend(batch_embedding)

chunk_matrix = np.array(vectors)
scores = chunk_matrix.dot(uq_vector)
highest_match_idx = np.argmax(scores)

print(
    "Q3: ", chunks[highest_match_idx]["filename"]
)  # 02-vector-search/lessons/07-sqlitesearch-vector.md


############################## Q4 ##############################
uq_4 = "What metric do we use to evaluate a search engine?"
uq_4_vector = embed.encode(uq_4)

v_index = VectorSearch(
    keyword_fields=["filename"],
)
v_index.fit(chunk_matrix, chunks)

top_results = v_index.search(uq_4_vector, num_results=1)
print("Q4: ", top_results[0]["filename"])  # 04-evaluation/lessons/05-search-metrics.md


############################## Q5 ##############################
uq_5 = "How do I store vectors in PostgreSQL?"
uq_5_vector = embed.encode(uq_5)

index = Index(keyword_fields=["filename"], text_fields=["content"])
index.fit(chunks)


top_vect_results = v_index.search(uq_5_vector, num_results=5)
top_vect_results_set = {r["filename"] for r in top_vect_results}

top_text_results = index.search(uq_5, num_results=5)
top_text_results_set = {r["filename"] for r in top_text_results}

print(
    "Q5: ", top_vect_results_set - top_text_results_set
)  # 02-vector-search/lessons/08-pgvector.md


############################## Q6 ##############################
# let's implement rrf = 1 / k + rank
def rrf(result_lists, k=60, num_results=5):
    """Reciprocal Rank Fusion"""
    scores = {}
    docs = {}
    for results in result_lists:
        for rank, doc in enumerate(results):
            chunk_id = (doc["filename"], doc["start"])  # serves as id for this chunk
            docs[chunk_id] = doc
            # increment the score if the item is in list
            new_score = scores.get(chunk_id, 0) + 1 / (k + rank)
            scores[chunk_id] = new_score

    # sort the scores dict on values, but return the key
    top_doc_keys = sorted(scores.keys(), key=scores.get, reverse=True)
    return [docs[key] for key in top_doc_keys[:num_results]]


# ask new question
uq_6 = "How do I give the model access to tools?"
uq_6_vector = embed.encode(uq_6)

v_scores = v_index.search(uq_6_vector, num_results=5)
i_scores = index.search(uq_6, num_results=5)

# rank fusion of both results
results = rrf([v_scores, i_scores])

print("Q6: ", results[0]["filename"])  # 01-agentic-rag/lessons/13-function-calling.md
