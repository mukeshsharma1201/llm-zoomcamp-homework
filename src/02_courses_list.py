import requests
import truststore

# Use the OS trust store (macOS keychain) so the corporate Zscaler root CA
# is trusted, just like it is in the browser. Must run before any requests.
truststore.inject_into_ssl()


docs_url = "https://datatalks.club/faq/json/courses.json"
response = requests.get(docs_url)
courses_list = response.json()


documents = []
url_prefix = "https://datatalks.club/faq"

for course in courses_list:
    course_url = f"""{url_prefix}{course["path"]}"""

    course_response = requests.get(course_url)
    course_response.raise_for_status()
    course_data = course_response.json()

    documents.extend(course_data)

print(documents)
