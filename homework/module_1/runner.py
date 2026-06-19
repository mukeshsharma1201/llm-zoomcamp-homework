from toyaikit.chat.runners import (
    OpenAIChatCompletionsRunner,
    ChatCompletionSystemMessageParam,
)


class GeminiChatCompletionsRunner(OpenAIChatCompletionsRunner):
    def _initialize_messages(
        self,
        previous_messages=None,
    ):

        if not previous_messages:
            return [
                ChatCompletionSystemMessageParam(
                    role="system",
                    content=self.developer_prompt,
                )
            ]

        return list(previous_messages)

    def loop(
        self,
        prompt,
        previous_messages=None,
        callback=None,
        output_format=None,
    ):

        result = super().loop(
            prompt=prompt,
            previous_messages=previous_messages,
            callback=callback,
            output_format=output_format,
        )

        return result
