from minsearch import Index

# from sqlitesearch import TextSearchIndex
from gitsource import GithubRepositoryDataReader
from gitsource import chunk_documents


def fetch_documents():
    reader = GithubRepositoryDataReader(
        repo_owner="DataTalksClub",
        repo_name="llm-zoomcamp",
        commit_id="8c1834d",
        allowed_extensions={"md"},
        filename_filter=lambda path: "/lessons/" in path,
    )

    files = reader.read()
    documents = []

    for file in files:
        doc = file.parse()
        documents.append(doc)

    return documents


def get_chunks(documents, size=2000, step=1000):
    chunks = chunk_documents(documents, size=size, step=step)
    return chunks


def build_index(documents):
    idx = Index(
        text_fields=["content"],
        keyword_fields=["filename"],
    )
    idx.fit(documents)
    return idx
