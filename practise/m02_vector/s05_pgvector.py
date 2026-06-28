import sys
from pathlib import Path

# Put the project root on sys.path so `from src....` works when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import truststore
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
import psycopg

from practise.config import get_db_config
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


# here, instead of creating the vector index in-memory or sqlite, we directly use postgres pgvector
db_cfg = get_db_config()
conn = psycopg.connect(db_cfg.uri)
# activate the pgvector extension
conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

# create postgres table to store embedding with vector datatype provided by the extension
conn.execute("""
    DROP TABLE IF EXISTS documents
""")

conn.execute("""
    CREATE TABLE documents(
        id SERIAL PRIMARY KEY,
        course TEXT,
        section TEXT,
        question TEXT,
        answer TEXT,
        embedding vector(384)
    )
""")


def vec_to_str(vector):
    return "[" + ",".join(str(x) for x in vector) + "]"


print(f"{'*' * 30} Inserting vectors into Postgres {'*' * 30}")
for doc, vec in tqdm(zip(documents, vectors), total=len(documents)):
    conn.execute(
        """
        INSERT INTO documents (course, section, question, answer, embedding)
        VALUES (%s, %s, %s, %s, %s::vector)
        """,
        (
            doc["course"],
            doc["section"],
            doc["question"],
            doc["answer"],
            vec_to_str(vec),
        ),
    )

conn.commit()  # save to db


############## manual searching ##############
query = "I just discovered the course. Can I still join it?"
query_vector = model.encode(query)
query_str = vec_to_str(query_vector)

# this is brute force search - no index
results = conn.execute(
    """
    SELECT course, question, answer,
           1 - (embedding <=> %s::vector) AS similarity
    FROM documents
    where course = %s
    ORDER BY (embedding <=> %s::vector)
    LIMIT 3
    """,
    (query_str, "llm-zoomcamp", query_str),
).fetchall()

for row in results:
    print(row)


############## search method override ##############
from openai import OpenAI
from practise.m01_agentic_rag.rag_helper import RAGBase
from practise.config import get_llm_provider_config


llm_cfg = get_llm_provider_config()
client = OpenAI(base_url=llm_cfg.base_url, api_key=llm_cfg.api_key)

# create index of embeddings
conn.execute("""
    CREATE INDEX ON documents
    USING hnsw (embedding vector_cosine_ops)
""")


class RAGPgVector(RAGBase):
    def __init__(self, embedder, conn, **kwargs):
        super().__init__(index=None, **kwargs)
        self.embedder = embedder
        self.conn = conn

    def search(self, query, num_results=5):
        """This is a Exact Nearest Neighbor search if we do not use a Index,
        as it will compare the query vector with all other vectors in the DB.
        AKA - Brute force search.
        With the Index created, it will use it and trade-off some accuracy with high speed.
        AKA - Approx Nearest Neighbor
        """
        query_vector = self.embedder.encode(query)
        query_str = vec_to_str(query_vector)

        rows = self.conn.execute(
            """
            SELECT course, section, question, answer
            FROM documents
            WHERE course = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (self.course, query_str, num_results),
        ).fetchall()

        return [
            {"course": r[0], "section": r[1], "question": r[2], "answer": r[3]}
            for r in rows
        ]


assistant = RAGPgVector(
    conn=conn,
    embedder=model,
    llm_client=client,
    model=llm_cfg.model,
)

print(f"{'*' * 30} RAG {'*' * 30}")
user_query = "its late, can I still sign up?"
print("Query: ", user_query)
answer = assistant.rag(user_query)
print(answer)

# close connection to the DB
conn.close()
