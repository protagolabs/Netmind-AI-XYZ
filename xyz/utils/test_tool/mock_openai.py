"""

==============
 Mock OpenAI API Client
==============
@file_name: mock_openai.py
@description: This module contains a mock implementation of the OpenAI API designed to simulate responses for chat completions.
The MockOpenAI class within this module can be used to develop and test applications that integrate with OpenAI's chat completions
without the need to make actual API calls. This helps in scenarios where API usage might be limited or costly, and during the
initial phases of development where frequent interaction with the OpenAI API is required.

## Initialization
The MockOpenAI class requires an API key upon initialization, which is simulated within this mock setup.
It initializes with default responses and configurations, which can be customized based on specific testing needs.

For detailed information on these parameters, refer to the OpenAI documentation. https://platform.openai.com/docs/api-reference/chat/create

## Methods
The class includes two primary methods for interacting with the simulated OpenAI:

- `create`: Simulates the creation of chat completions, supporting both standard and streaming outputs.
  Raises ValueError if the provided parameters do not meet expected formats or values.

- `stream_create`: Simulates the streaming of chat completions for testing streaming functionalities.

These methods simplify the process of integrating OpenAI functionalities into your applications, allowing for both
standard and streaming interactions.

## Motivation
The main motivation behind creating this mock class is to provide developers with a reliable and easy-to-use tool for
testing and developing applications that interact with the OpenAI chat completion API. It ensures that developers
can prototype and debug features without incurring API costs or hitting rate limits, fostering a smoother development
experience.

"""


from collections.abc import Mapping, Sequence
from typing import Final, NamedTuple, Optional

_DEFAULT_REPLY: Final[str] = 'RESPONSE:Hello\n'
_DEFAULT_SCORE: Final[float] = -0.1
_DEFAULT_STREAM_REPLY: Final[str] = 'STREAM_RESPONSE:HELLO\n'


class OpenAIMessage(NamedTuple):
    """
    Represents a message within the chat completion structure.

    Attributes
    ----------
    role : str
        The role of the entity sending the message (e.g., 'user', 'assistant').
    content : str
        The content of the message.
    """
    role: str
    content: str


class ChatCompletionTokenLogprob(NamedTuple):
    """
    Represents a single token and its associated log probability within a message.

    Attributes
    ----------
    token : str
        The token (word or punctuation) that was part of the response.
    logprob : float
        The log probability of the token, indicating how likely it was to occur.
    """
    token: str
    logprob: float


class ChoiceLogProbs(NamedTuple):
    """
    Contains a sequence of `ChatCompletionTokenLogprob` representing the tokens and their log probabilities for a choice.

    Attributes
    ----------
    content : Sequence[ChatCompletionTokenLogprob]
        A sequence of token log probability pairs for a particular completion.
    """
    content: Sequence[ChatCompletionTokenLogprob]


class Delta(NamedTuple):
    """
    Optional additional information about the change between two messages or states.

    Attributes
    ----------
    content : Optional[str]
        Textual content describing or quantifying the change, if any.
    """
    content: Optional[str] = None


class Choice(NamedTuple):
    """
    Represents a single choice or option provided by the completion system.

    Attributes
    ----------
    index : int
        The index of the choice among all options provided in the completion.
    message : OpenAIMessage
        The message associated with this choice.
    logprobs : ChoiceLogProbs
        Log probabilities for tokens in the message.
    finish_reason : str
        The reason why this part of the conversation or prompt was concluded.
    delta : Delta
        Optional additional data or context relevant to this choice.
    """
    index: int
    message: OpenAIMessage
    logprobs: ChoiceLogProbs
    finish_reason: str
    delta: Delta


class ChatCompletion(NamedTuple):
    """
    Encapsulates the overall completion response from a chat model.

    Attributes
    ----------
    choices : Sequence[Choice]
        A sequence of choices provided by the model. Each choice includes detailed information such as the message and associated probabilities.
    """
    choices: Sequence[Choice]


class MockOpenAI:
    """
    A mock class that simulates the behavior of the OpenAI API, supporting both standard and streaming chat completion features.

    This class is intended for testing and development purposes, allowing users to simulate responses from the OpenAI chat completion API without actual API calls.

    Attributes
    ----------
    api_key : str
        The API key used for authentication (simulated in this mock).

    Parameters
    ----------
    api_key : str
        A string representing a fake API key for the OpenAI API.

    Methods
    -------
    chat()
        Provides access to chat interface methods.

    completions()
        Provides access to the completion interface methods.

    create(model, messages, stream=False, **kwargs)
        Simulates the creation of chat completions, either in standard or stream mode.

    stream_create(choices)
        Simulates the creation of streamed chat completions.
    """

    def __init__(self, api_key: str):
        """
        Initializes the MockOpenAI instance with the provided API key.

        Parameters
        ----------
        api_key : str
            A string representing a fake API key for the OpenAI API.
        """
        self.api_key = api_key
        self._reply = _DEFAULT_REPLY
        self._stream_reply = _DEFAULT_STREAM_REPLY
        self._delta_ptr = 0
        self._delta_end = len(self._stream_reply)

    @property
    def chat(self):
        """
        Simulated chat interface property.

        Returns
        -------
        self
            Returns the instance itself, simulating the chat interface.
        """
        return self

    @property
    def completions(self):
        """
        Simulated completions interface property.

        Returns
        -------
        self
            Returns the instance itself, simulating the completions interface.
        """
        return self

    def create(
            self,
            model: str,
            messages: Sequence[Mapping[str, str]],
            stream: bool = False,
            **kwargs,
    ) -> list[ChatCompletion] | ChatCompletion:
        """
        Simulates the creation of chat completions based on input parameters, supporting both standard and streaming outputs.

        Parameters
        ----------
        model : str
            The model identifier, which in real use would determine the behavior of the completion.
        messages : list of dict
            A sequence of dictionaries representing messages in the conversation.
        stream : bool, optional
            If True, simulates streaming completions.
        **kwargs : dict
            Additional keyword arguments for simulating API parameters such as 'n' for the number of completions.

        Returns
        -------
        list of ChatCompletion or ChatCompletion
            Depending on the 'stream' parameter, returns a single ChatCompletion or a list of ChatCompletion objects.
        """
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
        if stream:
            self._delta_ptr += 1
            return [ChatCompletion(choices=choices)]
        return ChatCompletion(choices=choices)
