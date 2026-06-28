# LLM Zoomcamp

A personal collection of runnable scripts following the [DataTalksClub LLM Zoomcamp](https://github.com/DataTalksClub/llm-zoomcamp) course. It builds up a small **Retrieval-Augmented Generation (RAG)** pipeline step by step — from a single LLM call, to keyword search + RAG, to a tool-calling **agentic** loop, and on to **vector search** (in-memory, SQLite, pgvector, and a dependency-free ONNX embedder) — using an OpenAI-compatible client pointed at Google Gemini.

> This repo is **not** an installable package. It's a set of standalone, runnable course scripts. The project root is placed on `sys.path` (via a `.pth` file written into the virtualenv by `uv`, and via a small `sys.path.insert` shim at the top of each script) so the `practise.*` and `homework.*` imports resolve when you run a file directly.

---

## What's inside

```
llm-zoomcamp/
├── practise/                       # Course follow-along scripts
│   ├── config.py                   # Loads LLM provider + DB config from .env
│   ├── m01_agentic_rag/            # Module 1 — Agentic RAG
│   │   ├── s01_simple_call.py      # Plain LLM chat completion call
│   │   ├── s02_courses_list.py     # Fetch FAQ documents from DataTalksClub
│   │   ├── s03_search_and_rag.py   # Manual retrieve → augment → generate
│   │   ├── s04_use_rag_helper.py   # Same flow using the RAGBase helper
│   │   ├── s05_rag_agent_playground.py  # Hand-rolled tool-calling agent loop
│   │   ├── s06_rag_agent_framework.py   # Agent loop variant
│   │   ├── ingest.py               # Load FAQ data + build search indexes
│   │   └── rag_helper.py           # RAGBase: search → prompt → llm → answer
│   │
│   └── m02_vector/                 # Module 2 — Vector search
│       ├── s01_embedding.py        # Embed FAQ docs with sentence-transformers
│       ├── s02_vector_search.py    # In-memory vector search (minsearch) + RAG
│       ├── s03_vector_rag_ingest.py    # Persist embeddings to SQLite (sqlitesearch)
│       ├── s04_vector_rag_retrieve.py  # Load the SQLite index and run RAG
│       ├── s05_pgvector.py         # Embed + store/query in Postgres (pgvector)
│       ├── s06_onnx.py             # Embed with the local ONNX model (no torch)
│       └── onnx/
│           ├── download.py         # Pull the ONNX model from Hugging Face → models/
│           └── embedder.py         # Embedder: tokenize → onnxruntime → mean-pool
│
├── homework/                       # Course homework solutions
│   ├── config.py
│   ├── module_1/
│   │   ├── ingest.py               # Reads md lessons from the course GitHub repo
│   │   ├── rag_helper.py
│   │   ├── runner.py               # Gemini-compatible toyaikit runner
│   │   └── module_1_homework.py    # Q1–Q6 homework answers
│   └── module_2/
│       ├── module_2_homework.py    # Vector-search homework answers
│       └── onnx/                   # Local ONNX embedder (download.py + embedder.py)
│
├── Makefile                        # Spin pgvector Postgres up/down via Docker
├── main.py                         # Hello-world entrypoint
├── pyproject.toml                  # Dependencies (managed by uv)
├── faq.db                          # SQLite keyword index (generated)
├── faq_vectors2.db                 # SQLite vector index (generated, module 2)
├── models/                         # Local ONNX model cache (git-ignored)
├── .env.example                    # Template for required env vars
└── .python-version                 # Pins Python 3.12
```

### Concepts demonstrated

| Stage | Script | Idea |
|-------|--------|------|
| 1. Simple call | `m01_agentic_rag/s01_simple_call.py` | Call an LLM with a system + user message |
| 2. Ingest | `m01_agentic_rag/s02_courses_list.py`, `m01_agentic_rag/ingest.py` | Pull FAQ JSON / lesson markdown into documents |
| 3. RAG | `m01_agentic_rag/s03_search_and_rag.py`, `m01_agentic_rag/s04_use_rag_helper.py` | Retrieve relevant docs, build a grounded prompt, generate |
| 4. Agentic RAG | `m01_agentic_rag/s05_*`, `m01_agentic_rag/s06_*`, `module_1_homework.py` | Give the LLM a `search` tool and let it loop until it has an answer |
| 5. Embeddings | `m02_vector/s01_embedding.py` | Turn documents into dense vectors |
| 6. Vector search | `m02_vector/s02_vector_search.py` | Nearest-neighbour retrieval over embeddings (in-memory) |
| 7. Persisted vectors | `m02_vector/s03_*`, `s04_*` | Split ingest/retrieve, persist the index to SQLite |
| 8. pgvector | `m02_vector/s05_pgvector.py` | Store and query embeddings in Postgres with `pgvector` |
| 9. Local ONNX embedder | `m02_vector/s06_onnx.py`, `onnx/` | Embed without `torch` — tokenizers + `onnxruntime` only |

Keyword search is powered by [`minsearch`](https://pypi.org/project/minsearch/) (in-memory) and [`sqlitesearch`](https://pypi.org/project/sqlitesearch/) (persisted to `faq.db`). Vector search uses `minsearch`'s `VectorSearch`, `sqlitesearch`'s `VectorSearchIndex` (persisted to `faq_vectors2.db`), and [`pgvector`](https://github.com/pgvector/pgvector) on Postgres. Embeddings come from [`sentence-transformers`](https://www.sbert.net/) (`all-MiniLM-L6-v2`) or, for a torch-free path, a local ONNX export of the same model run via [`onnxruntime`](https://onnxruntime.ai/).

---

## Prerequisites

- **Python 3.12** (pinned in `.python-version`)
- **[uv](https://docs.astral.sh/uv/)** — the package/environment manager used here
- A **Google Gemini API key** (the scripts use an OpenAI-compatible client pointed at Gemini's endpoint)
- **Docker** — only for the pgvector example (`m02_vector/s05_pgvector.py`)

Install `uv` if you don't have it:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# or via Homebrew
brew install uv
```

---

## Installation

```bash
# 1. Clone and enter the repo
git clone https://github.com/mukeshsharma1201/llm-zoomcamp-homework.git llm-zoomcamp
cd llm-zoomcamp

# 2. Create the virtualenv and install all dependencies from uv.lock
uv sync

# 3. Create your .env from the template (see "Environment" below)
cp .env.example .env
# then edit .env and add your real API key
```

`uv sync` creates a `.venv/` in the project root and installs the locked dependency set. You don't need to activate the venv manually — prefix commands with `uv run` (recommended) — but you can activate it if you prefer:

```bash
source .venv/bin/activate
```

---

## Environment

The scripts read configuration from a `.env` file in the project root (loaded via `python-dotenv` in [practise/config.py](practise/config.py) and [homework/config.py](homework/config.py)).

Copy `.env.example` and fill in your values:

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `GOOGLE_API_KEY` | ✅ Yes | Your Gemini API key. **Never commit this.** | `AIza...` |
| `GOOGLE_BASE_URL` | ✅ Yes | OpenAI-compatible Gemini endpoint | `https://generativelanguage.googleapis.com/v1beta/openai/` |
| `LLM_MODEL` | ✅ Yes | Model name to call | `gemini-3.1-flash-lite` |
| `PG_CONN_STR` | ⚪ pgvector only | Postgres connection string for the pgvector example | `postgresql://user:pswd@localhost:5432/faq` |

Example `.env`:

```dotenv
GOOGLE_API_KEY=your-real-key-here
GOOGLE_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
LLM_MODEL=gemini-3.1-flash-lite
PG_CONN_STR=postgresql://user:pswd@localhost:5432/faq
```

Get a key from [Google AI Studio](https://aistudio.google.com/apikey). The free tier is enough to run these scripts (the homework script is written to be frugal with tokens).

> **Corporate network note:** `config.py` (and each network-touching script) calls `truststore.inject_into_ssl()` before any network requests, so the OS trust store (e.g. macOS Keychain) is used. This makes requests work behind a corporate TLS-inspecting proxy (like Zscaler) without extra certificate setup.

`.env` is git-ignored — only `.env.example` is committed.

---

## Running the scripts

Run any script with `uv run` so it uses the project's virtualenv:

### Module 1 — Agentic RAG

```bash
# 1. Simple LLM call
uv run python practise/m01_agentic_rag/s01_simple_call.py

# 2. Fetch the FAQ documents
uv run python practise/m01_agentic_rag/s02_courses_list.py

# 3. Manual RAG: retrieve → augment → generate
uv run python practise/m01_agentic_rag/s03_search_and_rag.py

# 4. RAG using the RAGBase helper (builds the SQLite index faq.db)
uv run python practise/m01_agentic_rag/s04_use_rag_helper.py

# 5. Agentic RAG (LLM with a search tool, looping until done)
uv run python practise/m01_agentic_rag/s05_rag_agent_playground.py

# Homework — Module 1 (prints answers to Q1–Q6)
uv run python homework/module_1/module_1_homework.py
```

### Module 2 — Vector search

```bash
# 1. Embed the FAQ docs with sentence-transformers
uv run python practise/m02_vector/s01_embedding.py

# 2. In-memory vector search + RAG (minsearch VectorSearch)
uv run python practise/m02_vector/s02_vector_search.py

# 3. Ingest: embed and persist the vector index to faq_vectors2.db
uv run python practise/m02_vector/s03_vector_rag_ingest.py

# 4. Retrieve: load the persisted index and answer a question
uv run python practise/m02_vector/s04_vector_rag_retrieve.py

# 5. pgvector — start Postgres first (see below), then:
uv run python practise/m02_vector/s05_pgvector.py

# 6. Torch-free embeddings via a local ONNX model
uv run python practise/m02_vector/onnx/download.py   # one-time: fetch model → models/
uv run python practise/m02_vector/s06_onnx.py

# Homework — Module 2
uv run python homework/module_2/onnx/download.py     # one-time: fetch the ONNX model
uv run python homework/module_2/module_2_homework.py
```

Most scripts have a question hard-coded near the bottom (under `if __name__ == "__main__":`) — edit it to ask your own.

#### pgvector with Docker

The pgvector example needs a Postgres instance with the `pgvector` extension. The `Makefile` wraps the Docker commands:

```bash
make up         # start a pgvector/pgvector:pg17 container on localhost:5432
make db-logs    # follow the container logs
make down       # stop and remove the container
make db-clear   # stop the container and delete its data volume
```

The container is created with user `user` / password `pswd` / database `faq` — match these in your `PG_CONN_STR`.

#### Local ONNX embedder

`onnx/download.py` pulls an ONNX export of `all-MiniLM-L6-v2` (the `Xenova/all-MiniLM-L6-v2` repo) from Hugging Face into `models/` (git-ignored). The `Embedder` in `onnx/embedder.py` then tokenizes with `tokenizers`, runs inference with `onnxruntime`, and mean-pools the token embeddings — so you can produce the same vectors without pulling in `torch`/`sentence-transformers` at runtime.

---

## Development

This project uses `ruff` for linting and formatting:

```bash
uv run ruff check .     # lint
uv run ruff format .    # format
```

---

## License

Personal learning project based on the DataTalksClub LLM Zoomcamp course materials.
