"""
============
OpenAIClient
============
@file_name: openai_client.py
@description:
This module provides an interface to interact with OpenAI services. We have encapsulated the functionality into a simple
-to-use class, enabling users to easily interface with OpenAI.

## Initialization
Users initialize the OpenAIClient by specifying the api_key and generate_args:

- `api_key`: The API key for OpenAI. This must be obtained from your OpenAI account.
- `generate_args`: Arguments for the chat completion request. For detailed information on these parameters, refer to the
 OpenAI documentation. https://platform.openai.com/docs/api-reference/chat/create

## Methods
The class includes two primary methods for interacting with OpenAI:

- `run`: This method is used to make standard requests to OpenAI. It accepts messages, images, and optionally tools as
parameters:
    - `messages`: A list of strings, each representing a conversation turn.
    - `images`: A list of URLs pointing to images to be included in the request.
    - `tools`: An optional list that specifies additional tools to be used in the request.
- `stream_run`: This method is designed for streaming requests to OpenAI and also requires the messages and images
parameters.
These methods simplify the process of integrating OpenAI functionalities into your applications, allowing for both
standard and streaming interactions.

## Motivation
Certainly, it's possible to use OpenAI's API directly, but by encapsulating it within a class, we can **simplify** the
process and make it more convenient for users to utilize OpenAI's services. Most importantly, this approach enhances
code reusability. Parameters commonly used in API requests can be encapsulated as arguments to class methods.
"""


__all__ = ["OpenAIClient"]


class OpenAIClient:
    """
    The OpenAI client which uses the OpenAI API to generate responses to messages.
    """

    def __init__(self, api_key=None, **generate_args):
        """Initializes the OpenAI Client.

        Parameters
        ----------
        api_key : str, optional
            The OpenAI API key.
        generate_args : dict, optional
            Arguments for the chat completion request.
            ref: https://platform.openai.com/docs/api-reference/chat/create
        """
        raise NotImplementedError

    def run(self, messages, tools = None,
            images = None):
        """
        Run the assistant with the given messages.

        Parameters
        ----------
        messages : list
            A list of messages to be processed by the assistant.
        tools : list, optional
            A list of tools to be used by the assistant, by default [].
        images : list, optional
            A list of image URLs to be used by the assistant, by default [].

        Returns
        -------
        str
            The assistant's response to the messages.

        Raises
        ------
        OpenAIError
            There may be different errors in different situations, which need to be handled according to the actual
                situation. An error message is printed in the console when an error is reported.
            ref: https://platform.openai.com/docs/guides/error-codes/python-library-error-types
        """
        raise NotImplementedError

    def stream_run(self, messages, images):
        """
        Run the assistant with the given messages in a streaming manner.

        Parameters
        ----------
        images : list
            A list of image URLs to be used by the assistant.
        messages : list
            A list of messages to be processed by the assistant.

        Yields
        ------
        str
            The assistant's response to the messages, yielded one piece at a time.

        Raises
        ------
        OpenAIError
            There may be different errors in different situations, which need to be handled according to the actual
                situation. An error message is printed in the console when an error is reported.
            ref: https://platform.openai.com/docs/guides/error-codes/python-library-error-types
        """
        raise NotImplementedError
