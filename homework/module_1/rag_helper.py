from textwrap import dedent


class RAGBase:
    # grounds the answer in our data and reduces hallucinations
    SYSTEM_PROMPT = dedent("""
    You're a course teaching assistant.
    You're given a question from a course student and your task is to answer it
    using course lessons provided in the context section.
                           
    Use the context to find relevant information and provide accurate
    answers. If the answer is not found in the context,
    respond with "I don't know.".
    If the context is empty, do not answer using your own knowledge.
    """).strip()

    # template for user questions
    USER_PROMPT = dedent("""
            User Question: {question}

            Context : 
            {context}
        """).strip()

    def __init__(
        self,
        index,
        llm_client,
        instructions=SYSTEM_PROMPT,
        prompt_template=USER_PROMPT,
        # course="llm-zoomcamp",
        model="gemini-3.1-flash-lite",
    ) -> None:
        self.index = index
        self.llm_client = llm_client
        self.instructions = instructions
        # self.course = course
        self.prompt_template = prompt_template
        self.model = model

    def search(self, query: str, num_results: int = 5) -> list[dict[str, str]]:
        """
        Search the course lessons for entries matching the given query
        """
        boost_dict = {"content": 3.0, "filename": 1}
        # filter_dict = {"course": self.course}

        return self.index.search(
            query,
            num_results=num_results,
            boost_dict=boost_dict,
            # filter_dict=filter_dict,
        )

    def build_context(self, retrieved_docs):
        lines = []
        for doc in retrieved_docs:
            _line = f"{str(doc['filename'])}\n{str(doc['content'])}"
            # _line = f"{str(doc['section']).strip()}\nQ: {str(doc['question']).strip()}\nA: {str(doc['answer']).strip()}\n"
            lines.append(_line)

        return "\n\n\n".join(lines)

    def build_prompt(self, query, search_results):
        context = self.build_context(search_results)
        return self.prompt_template.format(question=query, context=context)

    def llm(self, prompt):
        input_messages = [
            {"role": "system", "content": self.instructions},
            {"role": "user", "content": prompt},
        ]

        try:
            response = self.llm_client.chat.completions.create(
                model=self.model, messages=input_messages
            )
            return response
        except Exception as e:
            print(e)
            return "Something went wrong, unable to answer"

    def rag(self, query):
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        response = self.llm(prompt)
        return (response.choices[0].message.content, response.usage)
