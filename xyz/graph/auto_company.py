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

import logging
import inspect
import os
import time
import re
import json
from typing import Any

from xyz.node.agent import Agent
from xyz.elements.assistant.manager_assistant import ManagerAssistant
from xyz.elements.assistant.input_format_assistant import InputFormatAssistant
from xyz.utils.llm.openai_client import OpenAIClient


class AutoCompany(Agent):
    llm_client: OpenAIClient
    manager: ManagerAssistant
    agents: dict
    agents_info: str
    graph: dict

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

        self.graph = {}
        self.agents = {}
        self.agents_info = ""
        self.llm_client = llm_client
        self.manager = ManagerAssistant(llm_client)
        self.input_format_agent = InputFormatAssistant(llm_client)

        self.logger = self.create_logger(logger_path)

    def flowing(self, user_input, work_plan: dict = None) -> Any:
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

        # Step 1: Manager will analyze the task
        agents_info = self.get_agents_info()
        task_analysis = self.manager.analyze_task(user_input=user_input, agents_info=agents_info)
        self.logger.info("=======Start=========", extra={'step': "Task Analysis",
                                                         'agent': "Manager-Assistant"})
        task_analysis = self.stream_show(task_analysis)
        if "NO-WE-CAN-NOT" in task_analysis:
            return None

        # Step 2: Manager start to create work plan and distribute the work
        if work_plan is None:
            self.logger.info("=======Work-Plan=========", extra={'step': "Work Plan",
                                                                 'agent': "Manager-Assistant"})
            work_plan_str = self.manager.create_work_plan(task_analysis, agents_info)
            work_plan_str = self.stream_show(work_plan_str)
            work_plan = self.read_work_plan(work_plan_str)

        # Step 3: Manager start to execute the work plan
        solving_history = self.execute_work_plan(user_input=user_input, task=task_analysis, work_plan=work_plan)

        # Step 4: Manager do the summary
        summary_response = self.manager.summary(solving_history)
        self.logger.info("=======Summary=========", extra={'step': "Summary",
                                                           'agent': "Manager-Assistant"})
        summary_response = self.stream_show(summary_response)

        solving_record = ("User Input: " + user_input + "\n" + task_analysis
                          + solving_history + summary_response)

        self.logger.info("=======Finish=========\n\t\tSee you next time!!!", extra={'step': "Finish",
                                                                                    'agent': "Netmind_AI_XYZ"})

        return work_plan, solving_record

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

        working_history = ""

        # Prepare: Choose the start agent and end agent
        positions = {agent_info['position']: agent_info['name'] for agent_info in work_plan.values()}
        start_agent = positions.get('start')
        end_agent = positions.get('end')

        assert start_agent is not None, "No start agent found in the work plan"
        assert end_agent is not None, "No end agent found in the work plan"

        current_point = start_agent
        current_content = (f"The user input is: {user_input}\n\n"
                           f"The task analysis is: {task}\n\n"
                           f"The Plan is: \n\n{json.dumps(work_plan)}\n\n"
                           f"Now, we need let the first agent to start the work. "
                           f"We must call the first function, and get the parameters from the information above.")

        while current_point != "ErrorStop":

            self.logger.info("-------------", extra={'step': f"In Company Progress"
                                                             f": {work_plan[current_point]['sub_task']}",
                                                     'agent': f"Company Agent: {current_point}"})
            # Step 0: Get the agent object
            execute_agent = self.agents[current_point]

            # Step 1: Execute the agent
            self.logger.info("-------------\nI am communicating with this agent and arranging tasks for him. Please"
                             " wait."
                             "\n-------------", 
                             extra={'step': "Analysis the parameters", 'agent': "Manager-Assistant"})
            format_current_content = self.input_format_agent(input_content=current_content,
                                                             functions_list=[execute_agent.information])
            response = execute_agent(**format_current_content)
            current_response = self.stream_show(response)

            if work_plan[current_point]['position'] == "end":
                working_history += current_point + ":" + current_response + "\n\n"
                self.logger.info("The work plan is finished", extra={'step': "Finish",
                                                                     'agent': "None"})
                break

            # Step 2: Manager do the small summary
            next_list_info = self.get_next_list_info(work_plan[current_point])
            current_summary = self.manager.summary_step(working_history=working_history,
                                                        current_response=current_response,
                                                        next_list_info=next_list_info)

            # Step 3: Log the information
            self.logger.info("-------Step Summary------", extra={'step': f"Summarize this step",
                                                                 'agent': f"Manager-Assistant"})
            current_summary_content = self.stream_show(current_summary)

            # Step 4: Update the working history
            working_history += current_point + ":" + current_summary_content + "\n\n"
            self.input_format_agent.add_history([{"role": "assistant", "content": current_summary_content}])

            # Step 5: Update the current point
            next_name = self.get_special_part(pattern="next-employee", content=current_summary_content)
            name = json.loads(next_name)
            next_name = name['name']
            if next_name in work_plan:
                current_point = next_name
                current_content = current_response + self.get_special_part(pattern="next-step",
                                                                           content=current_summary_content)
            else:
                current_point = "ErrorStop"
                self.logger.info("This task is terminate with some error.", extra={'step': "Terminate",
                                                                                   'agent': "AutoSystem"})

        return working_history

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

        matches = self.get_special_part("working-plan", work_plan_str)
        working_graph = {}

        matches = re.sub(r'(?<!\\)\\(?!\\)', '\\\\\\\\', matches)
        agents = json.loads(matches)

        for i, agent in enumerate(agents):
            if i == 0:
                agent["position"] = "start"
            elif i == len(agents) - 1:
                agent["position"] = "end"
            else:
                agent["position"] = "in-progress"

            working_graph[agent["name"]] = agent

            if i != 0:
                working_graph[agents[i - 1]["name"]]["next"] = [agent["name"]]

        return working_graph

    def add_agent(self, agents: list) -> None:
        """
        Add the agents to the company. And you can use the agents to execute the work plan.

        Parameters
        ----------
        agents: list
            The list of the agents which you want to add to the company.
        """
        for agent in agents:
            self.agents[agent.information["function"]["name"]] = agent

    def get_agents_info(self):
        """
        Get the agents information which you have added to the company.

        Returns
        -------
        agents_info: str
            The agents information which you have added to the company.
        """

        agents_info = "In this company, we have the following agents:\n"

        for name, agent in self.agents.items():
            try:
                agents_info += (f"## ----------\nName: {name}\n"
                                f"Description: {agent.information['function']['description']}\n"
                                f"Input Type: {agent.input_type}\n"
                                f"Output Type{agent.output_type}\n## ----------\n\n")
            except KeyError or AttributeError:
                raise ValueError(f"The {name} agent must have the information and the input type and output type. as "
                                 f"required.")

        return agents_info

    def get_next_list_info(self, work_step: dict):
        """
        Get the next agents information which you have added to the company.

        Parameters
        ----------
        work_step: dict
            The element in the work plan, which is a dict store the information of the current agent.

        Returns
        -------
        next_info: str
            The next agents information which you have added to the company.
        """

        next_info = f"Next Agents: \n\n"
        for agent_name in work_step['next']:
            try:
                agent = self.agents[agent_name]
                next_info += (f"## ----------\nName: {agent_name}\n"
                              f"Description: {agent.information['function']['description']}\n"
                              f"Input Type: {agent.input_type}\n"
                              f"Output Type{agent.output_type}\n## ----------\n\n")
            except KeyError or AttributeError:
                raise ValueError(f"The {agent_name} agent must have the information and the input type and output type."
                                 f" as required.")

        return next_info

    @staticmethod
    def get_special_part(pattern: str, content: str) -> str:
        """
        Get the special part from the content by using the pattern. The special part must be in the `|||{pattern}`.

        Parameters
        ----------
        pattern: str
            The pattern which you want to extract from the content.
        content: str
            The full content which you want to extract the special part.

        Returns
        -------
        result: str
            The special part which you have extracted from the content.
        """

        pattern = "\|\|\|" + pattern
        pattern = pattern + '(.*?)' + pattern
        match = re.search(pattern, content, re.DOTALL)

        if match:
            result = match.group(1).strip()
        else:
            result = ""

        return result

    @staticmethod
    def create_logger(logger_path=None):
        """
        Create the logger for the company. And you can use the logger to log the information in the console and the file
        In fact, this is the user's command line UI.

        Parameters
        ----------
        logger_path: str
            The path of the logger file. If you don't provide the path, the logger will be saved in the logs folder.

        Returns
        -------
        logger: logging.Logger
            The logger which you can use to log the information in the console and the file.
        """

        class ColoredFormatter(logging.Formatter):
            RED = '\033[31m'
            CYAN = '\033[36m'
            GREEN = '\033[32m'
            RESET = '\033[0m'

            def format(self, record):
                record.step = self.RED + str(record.step) + self.RESET
                record.agent = self.CYAN + str(record.agent) + self.RESET
                record.msg = self.GREEN + str(record.msg) + self.RESET
                return super().format(record)

        if logger_path is None:
            current_time = time.strftime("%Y-%m-%d_%H-%M-%S")
            os.makedirs("logs", exist_ok=True)
            local_path = f"logs/tmp_log_{current_time}.log"
        else:
            local_path = logger_path
            # noinspection PyUnusedLocal
            logger_path = None

        PROCESS_LEVEL_NUM = 25
        logging.addLevelName(PROCESS_LEVEL_NUM, "PROCESS")

        def process(self, message, *args, **kws):
            if self.isEnabledFor(PROCESS_LEVEL_NUM):
                self._log(PROCESS_LEVEL_NUM, message, args, **kws)

        logging.Logger.process = process

        class StreamHandlerNoNewline(logging.StreamHandler):
            def emit(self, record):
                if record.levelno == PROCESS_LEVEL_NUM:
                    msg = record.getMessage()
                else:
                    # noinspection PyBroadException
                    try:
                        msg = self.format(record)
                        msg += "\n"
                    except:
                        msg = record.getMessage()
                        msg += "\n"
                if self.stream is None:
                    # noinspection PyUnresolvedReferences
                    self.stream = self._open()
                self.stream.write(msg)
                self.stream.flush()

        class FileHandlerNoNewline(logging.FileHandler):
            def emit(self, record):
                if record.levelno == PROCESS_LEVEL_NUM:
                    msg = record.getMessage()
                else:
                    # noinspection PyBroadException
                    try:
                        msg = self.format(record)
                        msg += "\n"
                    except:
                        msg = record.getMessage()
                        msg += "\n"
                if self.stream is None:
                    self.stream = self._open()
                self.stream.write(msg)
                self.stream.flush()

        # logger = logging.getLogger()
        logger = logging.getLogger("Assistant")
        logger.setLevel(logging.INFO)

        file_handler = FileHandlerNoNewline(local_path)
        file_handler.setLevel(logging.INFO)
        console_handler = StreamHandlerNoNewline()
        console_handler.setLevel(logging.INFO)

        formatter_console = ColoredFormatter('Step: %(step)s - Agent: %(agent)s - %(message)s')
        formatter_file = logging.Formatter('Step: %(step)s - Agent: %(agent)s - %(message)s')
        console_handler.setFormatter(formatter_console)
        file_handler.setFormatter(formatter_file)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    def stream_show(self, response):

        full_content = ""
        if inspect.isgenerator(response):
            for word in response:
                self.logger.process(word, extra={'step': "in progress", 'agent': "None"})
                full_content += word
        else:
            full_content = response
            self.logger.info(response, extra={'step': "in progress", 'agent': "None"})

        self.logger.process("\n", extra={'step': "in progress", 'agent': "None"})

        return full_content
