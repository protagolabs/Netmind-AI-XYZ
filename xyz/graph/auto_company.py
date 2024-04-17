"""
===========
AutoCompany
===========
@file_name: auto_company.py
@description:
This module implements a self-driving system for a small-scale multi-agent system. Users can manage multiple agents
through this system and automate the execution of work plans.

## Features
1. The system allows users to manage multiple agents and automate the execution of work plans, eliminating the need for
    manual configuration of multi-agent collaboration.
2. Once the agents are configured, the system automatically attempts to solve problems provided by the user.

## Usage
1. All agents must inherit from the Agent class:
    - They should be added to the company using the add_agent method.
    - Each agent must configure its information using the agent.set_information() method.
    - The set_input_type() method should be used to define the input format.
    - The set_output_type() method should be used to define the output format.
2. Execution can then proceed simply by calling auto_company(user_input).
3. All information will be recorded in a log file; if no log file path is provided, it will be stored in a logs folder
    in the current directory.
4. All information will also be displayed on the console.

## Motivation
1. We envision an AI-Society where, upon assigning a task, the system automatically schedules different agents and
    orchestrates their collective effort to execute the task autonomously.
2. This module represents our initial effort to automate the organization and execution of tasks by agents within the
    system.
3. We plan to continually enhance this system by incorporating features such as the automatic selection and scheduling
    of the most suitable agents from a large pool to carry out specific tasks.
"""


__all__ = ["AutoCompany"]

from xyz.node.agent import Agent
from xyz.utils.llm.openai_client import OpenAIClient


class AutoCompany(Agent):

    def __init__(self, llm_client: OpenAIClient, logger_path=None) -> None:
        """
        Initialize the AutoCompany. Which you can use to manage the agents and execute the work plan automatically.

        Parameters
        ----------
        llm_client: OpenAIClient
            The OpenAI client which you can use to communicate with the OpenAI API.
        logger_path: str
            The path of the logger file. If you don't provide the path, the logger will be saved in the `./logs` folder.
        """
        super().__init__()
        raise NotImplementedError

    def flowing(self, user_input, work_plan: dict = None):
        """
        The main function of the AutoCompany. Which you can use to manage the agents and execute the work plan
        automatically.
        And you can see the log in the console and the log file.

        Parameters
        ----------
        user_input: str
            The user input which you want to process.
        work_plan: dict or None
            The work plan which you want to execute. If you don't provide the work plan, the manager will generate
            the work plan automatically.

        Returns
        -------
        work_plan: dict
            The work plan which you have executed.
        solving_record: str
            The solving record which you have executed.
        """
        raise NotImplementedError

    def execute_work_plan(self, user_input: str, task: str, work_plan: dict):
        """
        Execute the work plan automatically.

        Parameters
        ----------
        user_input: str
            The user input
        task: str
            The task analysis
        work_plan: dict
            The work plan which you want to execute.

        Returns
        -------
        working_history: str
            The working history which you have executed.
        """
        raise NotImplementedError

    def read_work_plan(self, work_plan_str: str):
        """
        Read the work plan from the string. And return the work plan as a dict.

        Parameters
        ----------
        work_plan_str: str
            The work plan string.

        Returns
        -------
        working_graph: dict
            The work plan dict by using the json.
        """
        raise NotImplementedError

    def add_agent(self, agents: list) -> None:
        """
        Add the agents to the company. And you can use the agents to execute the work plan.

        Parameters
        ----------
        agents: list
            The list of the agents which you want to add to the company.
        """
        raise NotImplementedError

