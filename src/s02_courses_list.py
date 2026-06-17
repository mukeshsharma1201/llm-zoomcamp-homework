import requests
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


def load_documents():
    courses_list = get_courses_list()
    return get_course_documents(courses_list)


def main():
    documents = load_documents()
    print(documents[0])


if __name__ == "__main__":
    main()
