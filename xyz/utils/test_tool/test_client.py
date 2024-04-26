"""
============
TestClient
============
@file_name: test_client.py
@description:
This is a dummy LLM-CLient. It will not do any request. But it can check the request and response for openai API.

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
We are building a client for testing the OpenAI API. This client doesn't actually send requests to the OpenAI API 
but can inspect requests and responses. It's useful for validating whether an agent built using XYZ is functioning 
correctly.
"""

import os
import re
import httpx
import traceback
from typing import Generator, Union, Optional, Dict, List

from dotenv import load_dotenv
from openai import OpenAI
from openai import OpenAIError
from openai import Stream
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from typing_extensions import Literal

from mock_openai import MockOpenAI

__all__ = ["TestClient"]


class TestClient:
    """
    The OpenAI client which uses the OpenAI API to generate responses to messages.
    """
    client: OpenAI
    generate_args: dict
    last_time_price: float

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

        try:
            if api_key is None:
                load_dotenv()
                api_key = os.getenv('OPENAI_API_KEY')
            self.client = OpenAI(api_key=api_key)
        except OpenAIError:
            raise OpenAIError("The OpenAI client is not available. Please check the OpenAI API key.")

        # Set the default generate arguments for OpenAI's chat completions
        self.generate_args = {
            "model": "gpt-4-turbo",
            "temperature": 0.,
            "top_p": 1.0
        }
        # If the user provides generate arguments, update the default values
        self.generate_args.update(generate_args)
        self.validate_arguments()
        self.mock_openai = MockOpenAI(api_key)

    def validate_arguments(self):
        """
        Validate the generation argument in openai chat

        Raises
        ------
        ValueError
            If the generation argument is invalid.
        """
        valid_keys = {
            'model': Union[
                str,
                Literal[
                    "gpt-4-0125-preview",
                    "gpt-4-turbo-preview",
                    "gpt-4-1106-preview",
                    "gpt-4-vision-preview",
                    "gpt-4",
                    "gpt-4-0314",
                    "gpt-4-0613",
                    "gpt-4-32k",
                    "gpt-4-32k-0314",
                    "gpt-4-32k-0613",
                    "gpt-3.5-turbo",
                    "gpt-3.5-turbo-16k",
                    "gpt-3.5-turbo-0301",
                    "gpt-3.5-turbo-0613",
                    "gpt-3.5-turbo-1106",
                    "gpt-3.5-turbo-0125",
                    "gpt-3.5-turbo-16k-0613",
                ],
            ],
            "frequency_penalty": Optional[float],
            'temperature': Optional[float],
            'top_p': Optional[float],
            "max_tokens": Optional[int],
            'n': Optional[int],
            "logprobs": Optional[bool],
            'presence_penalty': Optional[float],
            'seed': Optional[int],
            'top_logprobs': Optional[int],
            'time_out': float | httpx.Timeout | None,
            "logit_bias": dict | None,
            'user': Optional[str],
            'stop': Optional[str] | List[str],
            # "functions":  list | None | NotGiven,
            # "function_call": completion_create_params.FunctionCall | None | NotGiven,
            # 'response_format': completion_create_params.ResponseFormat | None | NotGiven,
            # 'tool_choice': ChatCompletionToolChoiceOptionParam | None | NotGiven,
        }

        errors = []

        for key, value in self.generate_args.items():
            if key not in valid_keys:
                errors.append(f"Invalid argument: {key}")
            elif key == 'logit_bias':
                if not isinstance(value, dict):
                    errors.append(
                        f"Invalid type for {key}: Expected {dict}, got {type(value).__name__}")
                else:
                    if key == "logit_bias":
                        if not all(isinstance(k, str) for k in value.keys()) or not all(
                                isinstance(v, int) for v in value.values()):
                            errors.append(f"Invalid type in dictionary for {key}: Expected str keys and int values")
            else:
                expected_type = valid_keys[key]
                if not isinstance(value, expected_type):
                    errors.append(
                        f"Invalid type for {key}: Expected {expected_type}, got {type(value).__name__}")

        if errors:
            error_message = "Argument validation errors:\n" + "\n".join(errors)
            raise ValueError(error_message)

        for key, value in valid_keys.items():
            if key not in self.generate_args:
                self.generate_args[key] = None

    @staticmethod
    def validate_messages(messages: list):
        """
        Validate the messages

        Parameters
        ----------
        messages : list
            A list of messages to be processed by the assistant.

        Raises
        ------
        TypeError
            Raise if messages is not in a list, or each item in the list is not a dictionary in format
            {{"role": <role_name: str>, "content": <content: str>, 'name': <name: Optional[str]>}}
        """

        if not isinstance(messages, list):
            raise TypeError("Provided messages are not iterable. Please pass an iterable of message objects.")

        valid_roles = {'system', 'user', 'assistant', 'tool'}

        for item in messages:
            if not isinstance(item, dict):
                raise TypeError("Each message must be a dictionary.")

            if 'role' not in item or 'content' not in item:
                raise TypeError("Each message dictionary must contain 'role' and 'content' keys.")

            if item['role'] not in valid_roles:
                raise TypeError(f"Invalid 'role' value in message dictionary. Expected one of {valid_roles}.")

            if not isinstance(item['content'], str):
                raise TypeError("The 'content' value must be a string.")

            if 'name' in item:
                if not isinstance(item['name'], str):
                    raise TypeError("The 'name' value must be a string if provided.")
                if ' ' in item['name']:
                    raise TypeError("The 'name' value must not contain spaces.")

    @staticmethod
    def validate_tools(tools: Optional[list]):
        """
        Validate the Tools used in openai chat

        Parameters
        ----------
        tools : list, optional
            A list of tools to be used by the assistant, by default [].

        Raises
        ------
        TypeError
            Raise if tools is in invalid format.
        """

        if tools is None:
            return

        if not isinstance(tools, list):
            raise ValueError("Tools must be a list.")

        if len(tools) > 128:
            raise ValueError("A maximum of 128 functions are supported.")

        name_pattern = re.compile(r'^[a-zA-Z0-9_-]{1,64}$')

        for tool in tools:
            if not isinstance(tool, dict):
                raise ValueError("Each tool must be a dictionary.")

            if tool.get('type') != 'function':
                raise ValueError("Only 'function' type tools are supported.")

            function = tool.get('function')
            if not isinstance(function, dict):
                raise ValueError("The 'function' field must be a dictionary.")

            name = function.get('name')
            if not isinstance(name, str) or not name_pattern.match(name):
                raise ValueError(
                    "Function name must be a string of 1-64 characters, and can only include A-Z, a-z, 0-9, underscores, and dashes.")

            if 'description' in function:
                if not isinstance(function['description'], str):
                    raise ValueError("Function description must be a string.")

            if 'parameters' in function:
                parameters = function['parameters']
                if not isinstance(parameters, dict):
                    raise ValueError("Function parameters must be a dictionary.")

    def run(self,
            messages: List[Dict],
            tools: List = None,
            images: list = None) -> ChatCompletion | Stream[ChatCompletionChunk]:
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

        self.validate_arguments()
        self.validate_messages(messages=messages)
        if tools is not None:
            self.validate_tools(tools=tools)

        if images:
            last_message = messages.pop()
            text = last_message['content']
            content = [
                {"type": "text", "text": text},
            ]
            for image_url in images:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": image_url,
                    },
                })
            messages.append({
                "role": last_message['role'],
                "content": content
            })

        # If the user provides tools, use them; otherwise, this client will not use any tools
        if tools:
            tool_choice = "auto"
            local_tools = tools
            # noinspection PyUnusedLocal
            tools = None  # pyright: ignore[reportIncompatibleVariableOverride]
        else:
            local_tools = []
            tool_choice = "none"

        success_response = self.mock_openai.chat.completions.create(
            stream=False,
            messages=messages,
            tool_choice=tool_choice,
            tools=local_tools,
            **self.generate_args
        )
        return success_response

    def stream_run(self, messages: list, images: Optional[list[str]] = None) -> Generator[str, None, None]:
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

        self.validate_arguments()
        self.validate_messages(messages=messages)

        if images:
            last_message = messages.pop()
            text = last_message['content']
            content = [
                {"type": "text", "text": text},
            ]
            for image_url in images:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": image_url,
                    },
                })
            messages.append({
                "role": last_message['role'],
                "content": content
            })

        get_response_signal = False
        count = 0
        while not get_response_signal and count < 10:
            try:
                for response in self.mock_openai.chat.completions.create(
                        messages=messages,
                        stream=True,
                        timeout=5,
                        **self.generate_args
                ):
                    if response.choices[0].delta.content is None:
                        return None
                    else:
                        text = response.choices[0].delta.content
                        yield text
            except OpenAIError:
                error_message = str(traceback.format_exc())
                count += 1
                if count == 10:
                    raise OpenAIError(f"The error: {error_message}")
                print(f"The error: {error_message}")
                print(f"The messages: {messages}")
                time.sleep(2)


if __name__ == "__main__":
    test_client = TestClient(api_key="sk-1", top_p=0.8911, n=2, logit_bias={"s": 0})
    test_client.validate_arguments()
    mock_messages = [{"role": "assistant", "content": "yes", "name": "anc"}]
    mock_tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA"
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"]
                        }
                    },
                    "required": ["location"]
                }
            }
        }
    ]

    normal_response = test_client.run(messages=mock_messages, tools=mock_tools)
    print(normal_response)

    stream_response = test_client.stream_run(messages=mock_messages)
    import time

    for i in stream_response:
        print(i, end='', flush=True)
        time.sleep(0.1)
    print()
