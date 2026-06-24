import sys
from pathlib import Path

# Put the project root on sys.path so `from homework....` works when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from textwrap import dedent
from openai import OpenAI
from homework.config import get_llm_provider_config
from homework.module_1.ingest import fetch_documents, build_index, get_chunks
from homework.module_1.rag_helper import RAGBase


from homework.module_1.runner import GeminiChatCompletionsRunner
from toyaikit.llm import OpenAIChatCompletionsClient
from toyaikit.tools import Tools
from toyaikit.chat.interface import StdOutputInterface
from toyaikit.chat.runners import DisplayingRunnerCallback


############################## Q1 ##############################
documents = fetch_documents()
print("Q1: ", len(documents))  # 72


############################## Q2 ##############################
index = build_index(documents)
result = index.search(
    "How does the agentic loop keep calling the model until it stops?"
)
print("Q2: ", result[0]["filename"])  # 01-agentic-rag/lessons/14-agentic-loop.md

############################## Q3 ##############################
llm_cfg = get_llm_provider_config()
llm_client = OpenAI(base_url=llm_cfg.base_url, api_key=llm_cfg.api_key)
user_query = "How does the agentic loop keep calling the model until it stops?"

## commented to save Actual calls to llm - as I'm on free tier
rag_client = RAGBase(index=index, llm_client=llm_client, model=llm_cfg.model)
answer, usage = rag_client.rag(user_query)
print("Q3: ", usage.prompt_tokens)  # 7967

############################## Q4 ##############################
chunks = get_chunks(documents)
print("Q4: ", len(chunks))  # 295


############################## Q5 ##############################
ch_index = build_index(chunks)
rag_client = RAGBase(index=ch_index, llm_client=llm_client, model=llm_cfg.model)
answer, usage = rag_client.rag(user_query)
print("Q5: ", usage.prompt_tokens)  # 2619


############################## Q6 ##############################

ch_index = build_index(chunks)


def search(query: str, num_results: int = 5) -> list[dict[str, str]]:
    """
    Search the course lessons for entries matching the given query
    """
    print("search called")
    boost_dict = {"content": 3.0, "filename": 1}

    return index.search(
        query,
        num_results=num_results,
        boost_dict=boost_dict,
    )


SYSTEM_PROMPT = dedent("""
    You're a course teaching assistant.
    You're given a question from a course student and your task is to answer
    it using course lessons.

    If you want to look up information about course lesson, use the search function. 
    Use as many keywords from the user question as possible when making first request.

    Make multiple searches. First perform search, analyze the results 
    and then perform more searches. 

    The question has to be about the course, its contents or its logistics, off-topic questions 
    shouldn't be answered. If the search returns nothing, it's likely an off-topic question.
    If you can't answer the question course contents, don't do it yourself. Only use the 
    facts from search query.

    At the end, ask if there are other areas that the user wants to explore.
    """).strip()


agent_tools = Tools()
agent_tools.add_tool(search)


chat_interface = StdOutputInterface()
callback = DisplayingRunnerCallback(chat_interface)


runner = GeminiChatCompletionsRunner(
    tools=agent_tools,
    developer_prompt=SYSTEM_PROMPT,
    chat_interface=chat_interface,
    llm_client=OpenAIChatCompletionsClient(client=llm_client, model=llm_cfg.model),
)
result = runner.loop(
    prompt="How does the agentic loop work, and how is it different from plain RAG?",
    callback=callback,
)
