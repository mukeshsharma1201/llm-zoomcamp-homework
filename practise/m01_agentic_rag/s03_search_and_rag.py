import sys
from pathlib import Path

# Put the project root on sys.path so `from src....` works when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from textwrap import dedent  # noqa: E402

from minsearch import Index  # noqa: E402
from openai import OpenAI  # noqa: E402

from practise.config import get_llm_provider_config  # noqa: E402
from practise.m01_agentic_rag.s02_courses_list import load_documents  # noqa: E402


def retrieval(user_question):
    # data loading and indexing
    documents = load_documents()
    index = Index(
        text_fields=["section", "question", "answer"], keyword_fields=["course"]
    )
    index.fit(documents)

    # search / retrival
    search_results = index.search(
        user_question,
        boost_dict={"question": 2.0, "section": 0.5},
        filter_dict={"course": "llm-zoomcamp"},
        num_results=5,
    )

    return search_results


def context_builder(retrieved_docs):
    lines = []
    for doc in retrieved_docs:
        _line = f"{str(doc['section']).strip()}\nQ: {str(doc['question']).strip()}\nA: {str(doc['answer']).strip()}\n"
        lines.append(_line)

    return "\n".join(lines)


def prompt_builder(user_question, retrieved_docs):
    # grounds the answer in our data and reduces hallucinations
    SYSTEM_PROMPT = dedent("""
    You are a helpful course chat bot helper.
    Use the context to find relevant information and provide accurate
    answers. If the answer is not found in the context,
    respond with "I don't know."
    """).strip()

    context = context_builder(retrieved_docs)

    # template for user questions
    USER_PROMPT = (
        dedent("""
            User Question: {question}

            Context : 
            {context}
        """)
        .strip()
        .format(question=user_question, context=context)
    )

    return SYSTEM_PROMPT, USER_PROMPT


def llm_call(system_prompt, user_prompt):
    llm_cfg = get_llm_provider_config()
    client = OpenAI(api_key=llm_cfg.api_key, base_url=llm_cfg.base_url)
    _resp = client.chat.completions.create(
        model=llm_cfg.model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    try:
        print(f"Tokens Used {_resp.usage.total_tokens}")
        return _resp.choices[0].message.content
    except Exception as e:
        print(e)
        return "Something went wrong, unable to answer"


if __name__ == "__main__":
    user_question = "I just discovered the course. Can I join now?"

    # Retrieval - search for relevant sections in the course materials
    search_results = retrieval(user_question)

    # Augmentation - build a prompt for the LLM using the retrieved sections
    system_prompt, user_prompt = prompt_builder(user_question, search_results)

    # Generation - send the prompt to the LLM and get the answer
    response = llm_call(system_prompt, user_prompt)

    print(search_results)
    print(response)
