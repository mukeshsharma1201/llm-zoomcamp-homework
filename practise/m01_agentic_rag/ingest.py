from typing import Union

import requests
from minsearch import Index
from sqlitesearch import TextSearchIndex
import truststore

# Use the OS trust store (macOS keychain) so the corporate Zscaler root CA
# is trusted, just like it is in the browser. Must run before any requests.
truststore.inject_into_ssl()


def get_courses_list():
    docs_url = "https://datatalks.club/faq/json/courses.json"
    response = requests.get(docs_url)
    response.raise_for_status()
    return response.json()


def get_course_documents(courses_list):
    documents = []
    url_prefix = "https://datatalks.club/faq"

    for course in courses_list:
        course_url = f"""{url_prefix}{course["path"]}"""

        course_response = requests.get(course_url)
        course_response.raise_for_status()
        course_data = course_response.json()

        documents.extend(course_data)

    return documents


def load_faq_data(course: Union[str, None] = None):
    courses_list = get_courses_list()
    all_docs = get_course_documents(courses_list)
    if not course:
        return all_docs
    else:
        return [doc for doc in all_docs if doc["course"] == str(course)]


def build_index(documents):
    index = Index(
        text_fields=["section", "question", "answer"], keyword_fields=["course"]
    )
    index.fit(documents)

    return index


def build_text_index(documents, clear=False):
    index = TextSearchIndex(
        text_fields=["section", "question", "answer"],
        keyword_fields=["course"],
        db_path="faq.db",
    )
    if clear:
        index.clear()

    index.fit(documents)
    return index
