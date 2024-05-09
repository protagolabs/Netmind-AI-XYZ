"""
============
TestClient
============
@file_name: test_client.py
@description:
This is an openai TestClient. It will not do any request. But it can check the request and response for openai API.

## Initialization
Users initialize the OpenAIClient by specifying the api_key and generate_args:

- `api_key`: The API key for OpenAI. This must be obtained from your OpenAI account.
- `generate_args`: Default parameters for generating chat completions.
    The 'generate_args' attribute supports the following keys and types:
        - 'model': str,
            the model identifier, e.g., 'gpt-4-turbo'.
        - 'frequency_penalty': Optional[float]
            adjusts frequency of token usage to prevent repetition.
        - 'temperature': Optional[float]
            controls randomness in the response generation.
        - 'top_p': Optional[float]
            sets the nucleus sampling rate defining the probability mass used in token selection.
        - 'max_tokens': Optional[int]
            the maximum number of tokens to generate.
        - 'n': Optional[int]
            number of completions to generate for each prompt.
        - 'logprobs': Optional[bool]
            whether to return the log probabilities over the set of specified tokens.
        - 'presence_penalty': Optional[float]
            adjusts the likelihood of tokens based on their prior presence in the text.
        - 'seed': Optional[int]
            seed for random number generator for reproducibility.
        - 'top_logprobs': Optional[int]
            number of most likely tokens and their log probabilities to return.
        - 'time_out': float | httpx.Timeout | None
            the maximum duration to wait for the server response.
        - 'logit_bias': dict | None
            specifies biases for or against specific tokens during generation.
        - 'user': Optional[str]
            identifier for the user in the session.
        - 'stop': Optional[str] | List[str]
            tokens that signal the end of a generation.

For detailed information on these parameters, refer to the OpenAI documentation. https://platform.openai.com/docs/api-reference/chat/create

## Methods
The class includes two primary methods for interacting with OpenAI:

- `validate_arguments`: This function is used to validate the customized generation parameters.
Raises:
    - ValueError: If any generation arguments are invalid or of incorrect type.

- `get_gpt_models`:  Retrieves a list of available GPT models from the OpenAI API.
Raises:
    - If there is an error in communicating with the OpenAI API, this method will catch the OpenAIError and provide a pre-defined list of GPT model identifiers.

- `run`: This method is used to make standard requests to OpenAI. It accepts messages, images, and optionally tools as
parameters:
    - `messages`: A list of strings, each representing a conversation turn.
    - `images`: A list of URLs pointing to images to be included in the request.
    - `tools`: An optional list that specifies additional tools to be used in the request.

- `stream_run`: This method is designed for streaming requests to OpenAI and also requires the messages and images
parameters:
    - `messages`: A list of strings, each representing a conversation turn.
    - `tools`: An optional list that specifies additional tools to be used in the request.

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
import time
from typing import Generator, Optional, Dict, List

from dotenv import load_dotenv
from openai import OpenAI
from openai import OpenAIError
from openai import Stream
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from xyz.utils.test_tool.mock_openai import MockOpenAI

__all__ = ["TestClient"]


class TestClient:
    """
    A mock client to simulate interactions with the OpenAI API without making actual network requests.
    It allows for testing and validation of request and response handling, ensuring that integrations
    are correctly set up to use OpenAI functionalities.

    Attributes
    ----------
        client : OpenAI
            The OpenAI client instance used for API interactions.
        generate_args : dict
            Default parameters for generating chat completions.
        last_time_price : float
            Placeholder for tracking pricing, not implemented.
    """
    client: OpenAI
    generate_args: dict
    last_time_price: float

    def __init__(self, api_key=None, **generate_args):
        """
        A mock client to simulate interactions with the OpenAI API without making actual network requests.
        It allows for testing and validation of request and response handling, ensuring that integrations
        are correctly set up to use OpenAI functionalities.

        Parameters
        ----------
        api_key : str, optional
            The OpenAI API key.

        generate_args : dict, optional
            Default parameters for generating chat completions.
            The 'generate_args' attribute supports the following keys and types:
            - 'model': str,
                the model identifier, e.g., 'gpt-4-turbo'.
            - 'frequency_penalty': Optional[float]
                adjusts frequency of token usage to prevent repetition.
            - 'temperature': Optional[float]
                controls randomness in the response generation.
            - 'top_p': Optional[float]
                sets the nucleus sampling rate defining the probability mass used in token selection.
            - 'max_tokens': Optional[int]
                the maximum number of tokens to generate.
            - 'n': Optional[int]
                number of completions to generate for each prompt.
            - 'logprobs': Optional[bool]
                whether to return the log probabilities over the set of specified tokens.
            - 'presence_penalty': Optional[float]
                adjusts the likelihood of tokens based on their prior presence in the text.
            - 'seed': Optional[int]
                seed for random number generator for reproducibility.
            - 'top_logprobs': Optional[int]
                number of most likely tokens and their log probabilities to return.
            - 'time_out': float | httpx.Timeout | None
                the maximum duration to wait for the server response.
            - 'logit_bias': dict | None
                specifies biases for or against specific tokens during generation.
            - 'user': Optional[str]
                identifier for the user in the session.
            - 'stop': Optional[str] | List[str]
                tokens that signal the end of a generation.

            ref: https://platform.openai.com/docs/api-reference/chat/create

        """

        try:
            if api_key is None:
                load_dotenv()
                api_key = os.getenv('OPENAI_API_KEY')
            self.client = OpenAI(api_key=api_key)
        except OpenAIError:
            raise OpenAIError("The OpenAI client is not available. Please check the OpenAI API key.")

        self.generate_args = {
            "model": "gpt-4-turbo",
            "temperature": 0.,
            "top_p": 1.0
        }

        self.generate_args.update(generate_args)
        self.mock_openai = MockOpenAI(api_key)

    def validate_arguments(self):
        """
        Validates the generation arguments against expected types and values,
        ensuring all provided arguments meet API requirements.

        Raises
        ------
        ValueError
           If any generation arguments are invalid or of incorrect type.
        """

        valid_keys = {
            'model': (
                str, lambda x: (x in self.get_gpt_models(), f"is not a recognized model in {self.get_gpt_models()}")),
            "frequency_penalty": (Optional[float], lambda x: (-2.0 <= x <= 2.0, "must be between -2 and 2")),
            'temperature': (Optional[float], lambda x: (0.0 <= x <= 1.0, "must be between 0 and 1")),
            'top_p': (Optional[float], lambda x: (0.0 <= x <= 1.0, "must be between 0 and 1")),
            "max_tokens": (Optional[int], None),
            'n': (Optional[int], lambda x: (x >= 1, "must be at least 1")),
            "logprobs": (Optional[bool], None),
            'presence_penalty': (Optional[float], lambda x: (-2.0 <= x <= 2.0, "must be between -2 and 2")),
            'seed': (Optional[int], None),
            'top_logprobs': (Optional[int], None),
            'time_out': (float | httpx.Timeout | None, None),
            "logit_bias": (dict | None, lambda d: (
                all(isinstance(k, str) and isinstance(v, int) and -100 <= v <= 100 for k, v in d.items()),
                "keys must be strings and values must be integers, associated bias value must be from -100 to 100")),
            'user': (Optional[str], None),
            'stop': (Optional[str] | List[str], None),
        }

        errors = []
        for key, value in self.generate_args.items():
            if key not in valid_keys:
                errors.append(f"Invalid argument: {key}")
                continue

            expected_type, validator = valid_keys[key]
            if not isinstance(value, expected_type):
                expected_type_name = expected_type.__name__ if not isinstance(expected_type, tuple) else ', '.join(
                    t.__name__ for t in expected_type)
                errors.append(f"Invalid type for {key}: Expected {expected_type_name}, got {type(value).__name__}")

            if value is not None and validator:
                if callable(validator):
                    is_valid, error_msg = validator(value)
                    if not is_valid:
                        errors.append(f"Invalid value for '{key}': {value}. Value {error_msg}")

        if errors:
            error_message = "Argument validation errors:\n" + "\n".join(errors)
            raise ValueError(error_message)

    def get_gpt_models(self):
        """
        Retrieves a list of available GPT models from the OpenAI API.

        This method queries the OpenAI API to get a list of models that include
        'gpt' in their identifier. If the API call fails, a predefined list of
        GPT models is returned as a fallback.

        Returns
        -------
        list of str
            A list of model identifiers that include 'gpt'.

        Raises
        ------
        OpenAIError
            If there is an error in communicating with the OpenAI API, this
            method will catch the OpenAIError and provide a pre-defined list of GPT
            model identifiers.
        """
        try:
            models = self.client.models.list()
            openai_api_models = [m.id for m in models if 'gpt' in m.id]
        except OpenAIError:
            openai_api_models = ["gpt-4-0125-preview",
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
                                 'gpt-4-1106-preview',
                                 'gpt-3.5-turbo-1106',
                                 'gpt-3.5-turbo-16k',
                                 'gpt-4-turbo-2024-04-09',
                                 'gpt-4-turbo',
                                 'gpt-3.5-turbo-instruct-0914',
                                 'gpt-3.5-turbo-instruct',
                                 ]
        return openai_api_models

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
