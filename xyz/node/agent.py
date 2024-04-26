"""
=====
Agent
=====
@file_name: node.py
@description:
This module implements the foundational class for an AI-Agent. An AI-Agent is a callable entity designed to receive
input and return output. Users can create new AI-Agents by inheriting from this class. The flexibility of AI-Agent
allows users to inherit and implement their custom logic within the Agent.flowing() method, accommodating any
programming logic.

## Features of the AI-Agent include:
- Callable: AI-Agents can be invoked using the `agent()` method.
- Information Configuration: Users can set the properties of the AI-Agent using the `set_information()` method, adhering
    to the function call format specified by OpenAI.
- Nestability: AI-Agents can be nested within other AI-Agents.
- Structural Visibility: The structure of the AI-Agent can be inspected using the `__str__()` method, which can be
    directly printed using `print(agent)`.

## Motivation
AI-Agents leverage large language models and flexible programming combinations to achieve diverse functionalities.
The purpose of this class is to facilitate the encapsulation of custom AI-Agents for use within systems. AI-Agents also
provide foundational support for self-driving multi-agent systems. For detailed information, please refer to the
documentation and papers associated with this project.
"""

__all__ = ["Agent"]

from abc import abstractmethod
from typing import Callable, Any


class Agent:
    type: str
    information: dict
    output: dict

    def __init__(self):
        """
        The basic class of agent.
        User can create a new agent by inheriting this class.

        1. The user can create a new agent by inheriting this class.
        2. The user must set the information by call the method `set_information`. The format of the information is
        following the OpenAI's function call format. ref: https://platform.openai.com/docs/guides/function-calling
        3. The user must implement the method `flowing`. The method `flowing` is the main function of the agent.


        Users should create a new Agent by inheriting from this class.
        1. Users must set the Agent's information by calling the `set_information` method. The information format should
            adhere to the function call standards set by OpenAI.
        2. The `flowing()` method must be implemented. This method is the core functionality of the Agent.
        """

        self.llm_agents_dict = dict
        super().__setattr__("type", "agent")
        super().__setattr__("information", dict)
        super().__setattr__("output_type", str)
        super().__setattr__("input_type", "str")

    def _wrap_call(self, **kwargs) -> Callable:
        """
        The wrap call function for the agent. The user can call the agent by `agent(**kwargs)`.
        The agent will call the method `flowing` automatically.

        Parameters
        ----------
            **kwargs:
                The parameters of the agent are determined by the user-defined `flowing` method of the object.

        Returns
        -------
            self.flowing(**kwargs)
        """

        return self.flowing(**kwargs)

    __call__: Callable[..., Any] = _wrap_call

    @abstractmethod
    def flowing(self, **kwargs) -> Any:
        """
        The main function of the agent. The user must implement this function.

        Parameters
        ----------
            **kwargs:
                The parameters of the agent are determined by the user-defined `flowing` method of the object.
        """
        ...

    def set_generate_args(self, llm_agent_name: str = None, **generate_args: dict) -> None:
        """Set the generate arguments for the llm agent.
        If the llm_agent_name is None, the function will reset the generate arguments for all llm agents in this agent.
        If the llm_agent_name is not None, the function will reset the generate arguments for the specific llm agent.

        Parameters
        ----------
        llm_agent_name : str, optional
            The attribute name in this agent, by default None
            e.g. self.llm_1 = LLMAgent()
                The llm_agent_name is 'llm_1'
        generate_args : dict
            The generate arguments for the llm agent.
        """

        self.llm_agents_dict = {}
        # Update the llm agents in this agent.
        for key, value in vars(self).items():
            try:
                if "type" in vars(value) and value.type == "llm_agent":
                    # noinspection PyProtectedMember
                    self.llm_agents_dict[key] = value
            except:
                pass

        if llm_agent_name:
            assert llm_agent_name in self.llm_agents_dict, f"The llm agent {llm_agent_name} is not in the agent."
            self.llm_agents_dict[llm_agent_name].set_generate_args(**generate_args)
        else:
            for _name, llm_agent in self.llm_agents_dict.items():
                llm_agent.set_generate_args(**generate_args)

    def reset_generate_args(self, llm_agent_name: str = None) -> None:
        """Reset the generate arguments be the same with the OpenAIClient's generate args for the llm agent.
        If the llm_agent_name is None, the function will reset the generate arguments for all llm agents in this agent.
        If the llm_agent_name is not None, the function will reset the generate arguments for the specific llm agent.

        Parameters
        ----------
        llm_agent_name : str, optional
            The attribute name in this agent, by default None
            e.g. self.llm_1 = LLMAgent()
                The llm_agent_name is 'llm_1'
        """

        self.llm_agents_dict = {}
        # Update the llm agents in this agent.
        for key, value in vars(self).items():
            try:
                if "type" in vars(value) and value.type == "llm_agent":
                    # noinspection PyProtectedMember
                    self.llm_agents_dict[key] = value
            except:
                pass

        if llm_agent_name:
            assert llm_agent_name in self.llm_agents_dict, f"The llm agent {llm_agent_name} is not in the agent."
            self.llm_agents_dict[llm_agent_name].reset_generate_args()
        else:
            for _name, llm_agent in self.llm_agents_dict.items():
                llm_agent.reset_generate_args()

    def set_information(self, information: dict) -> None:
        """
        Set the information of the agent. And check the format of the information.

        Parameters
        ----------
            information: dict
                The information of the agent. The format of the information is following the OpenAI's function call
                format.

        Returns
        -------
            None
        """

        assert type(information) is dict, "The information must be a dict."
        assert "type" in information, "The information must have a key 'name'."
        assert "function" in information, "The information must have a key 'function'."
        assert "name" in information["function"], "The information must have a key 'name'."
        assert "description" in information["function"], "The information must have a key 'description'."
        assert "parameters" in information["function"], "The information must have a key 'parameters'."
        assert "type" in information["function"]["parameters"], "The information must have a key 'type'."
        assert "properties" in information["function"]["parameters"], "The information must have a key 'properties'."
        assert "required" in information["function"]["parameters"], "The information must have a key 'required'."
        for parameter in information["function"]["parameters"]["required"]:
            assert parameter in information["function"]["parameters"]["properties"], ("The required parameter must in "
                                                                                      "properties.")

        self.information = information

    def __str__(self) -> str:
        """
        To display the structure of this Agent, including any internally nested Agents, use the `__str__()` method.
            This method provides a textual representation of the Agent and its hierarchical structure, showing all
            nested components.

        Returns
        -------
            str
        """

        return self._structure()

    def _structure(self, order: int = 0) -> str:
        """
        To display the structure of this Agent, including any internally nested Agents, use the `__str__()` method.
            This method provides a textual representation of the Agent and its hierarchical structure, showing all
            nested components.

        Parameters
        ----------
        order: int
            The current nesting level of the Agent.

        Returns
        -------
            str
                The structure format of the agent.
        """

        pre_blank = "\t" * order
        info = (f"{pre_blank}Agent(name={self.information['name']}, description={self.information['description']}, "
                f"parameters={self.information['parameters']})")

        for key, value in vars(self).items():
            try:
                if "type" in vars(value) and (value.type == "agent" or value.type == "llm_agent"):
                    # noinspection PyProtectedMember
                    info += f"\n{pre_blank}[SubAgent: {key}: {value._structure(order + 1)}]"
            except:
                pass

        return info
