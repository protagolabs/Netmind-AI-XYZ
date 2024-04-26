from collections.abc import Mapping, Sequence
from typing import Final, NamedTuple, Optional

_DEFAULT_REPLY: Final[str] = 'RESPONSE:Hello'
_DEFAULT_SCORE: Final[float] = -0.1
_DEFAULT_STREAM_REPLY: Final[str] = 'STREAM_RESPONSE:HELLO'


class OpenAIMessage(NamedTuple):
    role: str
    content: str


class ChatCompletionTokenLogprob(NamedTuple):
    token: str
    logprob: float


class ChoiceLogProbs(NamedTuple):
    content: Sequence[ChatCompletionTokenLogprob]


class Delta(NamedTuple):
    content: Optional[str] = None


class Choice(NamedTuple):
    index: int
    message: OpenAIMessage
    logprobs: ChoiceLogProbs
    finish_reason: str
    delta: Delta


class ChatCompletion(NamedTuple):
    choices: Sequence[Choice]


class MockOpenAI:
    """A mock of the OpenAI class that provides a chat.completions.create method, now with streaming support."""

    stream: bool

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._reply = _DEFAULT_REPLY
        self._stream_reply = _DEFAULT_STREAM_REPLY
        self._delta_ptr = 0
        self._delta_end = len(self._stream_reply)

    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    def stream_create(self, choices):
        if self.stream:
            yield ChatCompletion(choices=choices)
            return None

    def create(
            self,
            model: str,
            messages: Sequence[Mapping[str, str]],
            stream: bool = False,
            **kwargs,
    ) -> list[ChatCompletion] | ChatCompletion:
        del model, messages
        samples = 1
        if 'n' in kwargs:
            samples = kwargs['n']

        choices = [
            Choice(
                index=i,
                message=OpenAIMessage(role='assistant', content=self._reply),
                logprobs=ChoiceLogProbs(
                    content=[
                        ChatCompletionTokenLogprob(
                            token='a', logprob=-0.1
                        )
                    ]
                ),
                finish_reason='stop',
                delta=Delta(content=_DEFAULT_STREAM_REPLY[self._delta_ptr]) if self._delta_ptr < self._delta_end else Delta(content=None)
            )
            for i in range(samples)
        ]
        self.stream = stream
        if stream:
            self._delta_ptr += 1
            return [ChatCompletion(choices=choices)]
        return ChatCompletion(choices=choices)
