import json
import sys
from pathlib import Path

# Put the project root on sys.path so `from src....` works when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from textwrap import dedent  # noqa: E402

from minsearch import Index  # noqa: E402
from openai import OpenAI  # noqa: E402

from practise.config import get_llm_provider_config  # noqa: E402
from practise.m01_agentic_rag.s02_courses_list import load_documents  # noqa: E402

SYSTEM_PROMPT = dedent("""
    You're a course teaching assistant.
    You're given a question from a course student and your task is to answer it.

    If you want to look up information, use the search function. 
    Use as many keywords from the user question as possible when making first requests.

    Make multiple searches. First perform search, analyze the results 
    and then perform more searches. 

    The question has to be about the course or its logistics, offtopic questions 
    shouldn't be answered. If the search returns nothing, it's likely an off-topic question.
    If you can't answer the question using FAQ, don't do it yourself. Only use the 
    facts from the FAQ database.

    At the end, ask if there are other areas that the user wants to explore.
    """).strip()


search_tool = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Seaerch the FAQ database of the course for entries matching the given query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query text to look up in the course FAQ.",
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}


def search(query):
    # data loading and indexing
    documents = load_documents()
    index = Index(
        text_fields=["section", "question", "answer"], keyword_fields=["course"]
    )
    index.fit(documents)

    # search / retrival
    search_results = index.search(
        query,
        boost_dict={"question": 2.0, "section": 0.5},
        filter_dict={"course": "llm-zoomcamp"},
        num_results=5,
    )

    return search_results


def make_tool_call(tool_calls):
    # supporting 1 call a time for now
    call = tool_calls[0]

    if call.function.name == "search":
        tool_args = json.loads(tool_calls[0].function.arguments)
        search_results = search(**tool_args)
        return {
            "role": "tool",
            "tool_call_id": call.id,
            "content": json.dumps(search_results, indent=2),
        }


def agent_loop(user_question, system_instructions=SYSTEM_PROMPT) -> str:
    llm_cfg = get_llm_provider_config()
    client = OpenAI(api_key=llm_cfg.api_key, base_url=llm_cfg.base_url)

    itrn = 1

    msg_history: list[dict[str, object]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_question},
    ]

    while True:
        print(f"Iteration {itrn} ...")

        has_tool_call = False

        # make llm call
        resp = client.chat.completions.create(
            model=llm_cfg.model,
            messages=msg_history,
            tools=[search_tool],
        )

        # append assistant message to history
        msg_history.append({
            "role": resp.choices[0].message.role,
            "content": resp.choices[0].message.content,
            "tool_calls": resp.choices[0].message.tool_calls,
        })

        # if llm wants to use tools - use tool
        tool_calls = resp.choices[0].message.tool_calls
        if tool_calls:
            has_tool_call = True
            tool_output = make_tool_call(tool_calls)  # in message formats

            # append message history
            msg_history.append(tool_output)
            # continue # -> send output back to llm
        elif resp.choices[0].message.content:
            return str(resp.choices[0].message.content)

        else:
            print("Could not understand what llm is saying...")
            print(resp)
            return "Could not understand what llm is saying..."

        # Run untill LLM does not want to make anyh tool calls
        itrn += 1
        if not has_tool_call:
            break
    # Loop broke unexpectedly
    return "Sorry, LLM could not answer..."


if __name__ == "__main__":
    user_question = "how to install olama?"
    print("USER: ", user_question)

    answer = agent_loop(user_question)

    print("ASSISTANT: ", answer)
