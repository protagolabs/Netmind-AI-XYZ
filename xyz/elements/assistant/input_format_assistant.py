""" 
====================
InputFormatAssistant
====================
@file_name: input_format_assistant.py
@description:
This module leverages information from the tool list available through the OpenAI API, assisting users in interfacing
messages using the function call format.

## Usage Requirements
Provide a list of information formatted according to OpenAI's Function Call specifications, along with the output from
the previous node, to obtain the parameter dictionary for the next node.

## Application Scenario
Currently, this module is primarily used within the AutoCompany project to facilitate the input and output coordination
    among different Agents.
"""

__all__ = ["InputFormatAssistant"]

import json

from xyz.node.agent import Agent
from xyz.node.basic.llm_agent import LLMAgent
from xyz.utils.llm.openai_client import OpenAIClient


class InputFormatAssistant(Agent):
    information: str
    llm_input_format: LLMAgent

    def __init__(self, llm_client: OpenAIClient) -> None:
        """
        The InputFormatAssistant is a tool to help user using the function calling format to interface the messages.
        This assistant will call for the OpenAI API and use a tools list which is some callables object's information,
        to get the parameters dict for the next callable object which user want to use.

        Parameters
        ----------
        llm_client: OpenAIClient
            The core agent for the assistant, which can call the OpenAI API.
        """
        super().__init__()

        # Set the information of the assistant. The information is used to help the user understand the assistant.
        self.set_information({
            "type": "function",
            "function": {
                "name": "InputFormatAssistant",
                "description": "Help user using the function calling format to interface the messages.",
                "parameters": {
                    "type": "object",
                    "properties": {"last_node_input": {"type": "string", "description": "The input of the last node."},
                                   "next_nodes_format": {"type": "list", "description": "The format of the next nodes."}
                                   },
                    "required": ["last_node_input", "next_nodes_format"],
                }
            },
        })

        self.messages = []
        # Using the template we designed to define the assistant, which can do the main task.
        self.llm_input_format = LLMAgent(template=input_format_prompts, llm_client=llm_client, stream=False)

    def flowing(self, input_content: str, functions_list: list, repeat_time: int = 0) -> dict:
        """
        The main function of the assistant, which can help user using the function calling format to interface
        the messages.

        Parameters
        ----------
        repeat_time : int
            The repeat time of the assistant.
        input_content: str
            The input of the last node.
        functions_list: list
            The list of OpenAI's Function call format information for some callables object.

        Returns
        -------
        dict
            The parameters dict for the next callable object which user want to use.
        """

        if repeat_time == 3:
            raise Exception("The manager assistant failed to distribute the tasks.")
        # noinspection PyBroadException
        try:
            completion = self.llm_input_format(messages=self.messages, input_content=input_content,
                                               tools=functions_list)
            parameters = completion.arguments
        except Exception:
            return self.flowing(input_content=input_content, functions_list=functions_list, repeat_time=repeat_time + 1)

        return json.loads(parameters)

    def add_history(self, messages: list) -> None:
        """
        Add messages to the assistant's global conversation history.

        Parameters
        ----------
        messages: list
            The list of messages to be added to the assistant's global conversation history.
        """
        self.messages.extend(messages)


input_format_prompts = [
    {"role": "system", "content": """ 
Now, you are a work liaison and you need to assist with work communication.

Your task is to use tools to understand the natural language information at work, and translate it into a function 
calling format for output to the next worker. This helps the next person on the job understand the job better.

Requirements:
1. You will receive a natural language message as input.
2. You must use tools to understand this natural language information.
3. You must fully understand what the parameters of tools mean.
4. You must convert the natural language information into the function calling format and use the input information to 
define various parameters in the tool.
5. You have a global conversation history that you can use in this task. Sometimes the parameter information for the 
function call you want to set is in the history dialog.

You are not allowed to fail, please be patient to complete this task.
Take a deep breath and start your work.
"""
     },

    {"role": "user", "content": """
The input in this time is : 
{input_content},
please choose a tool to interface this input.
Note: It is possible that the parameter information of the tool is in the history dialog. You can view the history 
dialog to get the parameter information of the tool. Please think carefully.
"""
     }
]
