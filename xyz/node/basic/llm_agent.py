""" 
========
LLMAgent
========
@file_name: llm_agent.py
@description:

This module implements an agent that processes messages using an LLM (Language Learning Model). Its primary function is
to process messages with a predefined template and then utilize various language models for further processing.

## Features of the LLMAgent include:
1. Template Usage: The agent utilizes a template, which is a list of strings formatted in OpenAI’s message structure.
    The text content within this template can be dynamically formatted using f-strings.
2. LLM Client: The agent includes a llm_client for invoking language model APIs. **Currently**, we use the OpenAI API.
    Users can also create their own LLM API by emulating the code found in `utils.llm.openai_client`.
3. Streaming Option: The agent has a stream parameter that controls whether the assistant’s messages are processed in a
    streaming manner.
4. Original Response Control: There is an original_response parameter that determines whether to return the raw
    response. If original_response is set to True, the raw response is returned; otherwise, only the content part is
        returned.

## Motivation
When creating a llm based agent, we need to pass messages and tools to the API. However, these items often require
    preprocessing before being sent. This agent facilitates this process, simplifying the engineering of prompts.
"""


from xyz.node.agent import Agent
from xyz.utils.llm.openai_client import OpenAIClient

__all__ = ["LLMAgent"]


class LLMAgent(Agent):
    """ 
    An assistant that uses the LLM (Language Learning Model) for processing messages.
    """

    def __init__(self, template: list, llm_client: OpenAIClient,
                 stream: bool = False, original_response: bool = False) -> None:
        # noinspection PyUnresolvedReferences
        """
        Initialize the assistant with the given template and core agent.

        Parameters
        ----------
        template: list
            The template for the assistant's prompts. It should be a list of OpenAI's messages.
        llm_client: OpenAIClient
            The core agent for the assistant.
        stream: bool, optional
            Whether to stream the assistant's messages, by default False.
        original_response: bool, optional
            Whether to return the original response, by default False.
        """
        raise NotImplementedError

    def flowing(self, messages: list = None,
                tools: list = None,
                images: list = None, **kwargs) -> str | Generator[str, None, None]:
        """When you call this assistant, we will run the assistant with the given keyword arguments from the prompts.
        Before we call the OpenAI's API, we do some interface on this message.

        Parameters
        ----------
        messages: list, optional
            The messages to use for completing the prompts, by default None.
        tools: list, optional
            The tools to use for completing the prompts, by default None.
        images: list, optional
            The images to use for completing the prompts, by default None.
        **kwargs
            The placeholders in the templates' text. They will be used to complete the prompts.

        Returns
        -------
        str/generator
            The response from the assistant. If stream == True, we will return a generator.
        """

        raise NotImplementedError

    def request(self, messages: list, tools: list, images: list):
        """
        Run the assistant with the given messages tools and images.
        """
        raise NotImplementedError

    def _stream_run(self, messages: list, images: list):
        """
        Run the assistant in a streaming manner with the given messages or images.

        Parameters
        ----------
        messages: list
            The messages which be used for call the LLM API.

        Returns
        -------
        generator
            The generator for the token(already be decoded) in assistant's messages.
        """
        raise NotImplementedError

    def debug(self):
        """
        Reset the assistant's messages.

        Returns
        -------
        dict
            The last time the request messages, tools and images.
        """

        raise NotImplementedError
