"""
================
ManagerAssistant
================
@file_name: manager_assistant.py
@description:
This is a demonstration of a tool designed to manage task allocation and oversee execution. It showcases how we can use
an Assistant to implement a small-scale Multi-Agent System that operates autonomously.

## Features
1. Task Analysis: manager.analyze_task(task_information)
2. Work Plan Development: manager.create_work_plan(task_analysis, agents)
3. Work Step Summary: manager.summary_step()
4. Work History Summary: manager.summary()

## Usage
This Manager is instantiated within the AutoCompany class and takes on the responsibility of task allocation and
execution supervision. It is an essential component for the normal operation of the AutoCompany.
"""

__all__ = ["ManagerAssistant"]

from typing import Generator

from xyz.node.agent import Agent
from xyz.node.basic.llm_agent import LLMAgent
from xyz.utils.llm.openai_client import OpenAIClient


class ManagerAssistant(Agent):
    information: str
    llm_prompt_engineer: LLMAgent
    original_task: str

    def __init__(self, llm_client: OpenAIClient) -> None:
        """
        The manager assistant is a class for manager task assignment and execution supervision.

        Parameters
        ----------
        llm_client: OpenAIClient
            For calling the OpenAI API to generate the response.
        """
        super().__init__()

        # Set the information of the assistant. The information is used to help the user understand the assistant.
        self.set_information({
            "type": "function",
            "function": {
                "name": "ManagerAssistant",
                "description": "Manager can make a plan for the task which contain task assignment and execution "
                               "supervision.",
                "parameters": {
                    "type": "object",
                    "properties": {"task": {"type": "str", "description": "The task which the user want to do."},
                                   "source": {"type": "list", "description": "The Agents in this company."},
                                   },
                    "required": ["task"],
                }
            },
        })

        # Using the template we designed to define the assistant, which can do the main task.
        self.llm_task_analysis = LLMAgent(template=task_analysis_prompt, llm_client=llm_client, stream=True)
        self.llm_work_plan_create = LLMAgent(template=work_plan_create_prompt, llm_client=llm_client, stream=True)
        self.llm_step_summary = LLMAgent(template=step_summary_prompt, llm_client=llm_client, stream=True)
        self.llm_summary = LLMAgent(template=summary_prompt, llm_client=llm_client, stream=True)
        self.llm_dynamic_select = LLMAgent(template=dynamic_select_prompt, llm_client=llm_client, stream=True)

    def flowing(self, task: str) -> str:
        """

        Parameters
        ----------
        task
            The task which the user want to do.

        Returns
        -------
        str
            Just the help information of the assistant. This method will not do the task. But it will set the task to
            the assistant.
        """

        self.original_task = task
        return "This manager assistant can help you to make a plan for the task which contain task assignment and " \
               "1. Analysis the task. manager.analyze_task(task_information)\n" \
               "2. Create a work plan. manager.create_work_plan(task_analysis, agents)\n" \
               "3. Summary the work plan step. manager.summary_step()\n" \
               "4. Summary the work history. manager.summary()\n"

    def analyze_task(self, user_input: str, agents_info: str) -> Generator:
        """
        The manager assistant can help the manager to analyze the task.

        Parameters
        ----------
        user_input: str
            The task information.
        agents_info: str
            The information of the agents which in this company.

        Returns
        -------
        Generator
            The analysis result of this task.
        """
        return self.llm_task_analysis(user_input=user_input,
                                      agents_info=agents_info)

    def create_work_plan(self, task_analysis: str, agents_info: str) -> Generator:
        """
        To make a work plan for the task.

        Parameters
        ----------
        task_analysis: str
            The analysis result of the task.
        agents_info: str
            The information of the agents which in this company.

        Returns
        -------
        Generator
            The work plan for the task.
        """
        return self.llm_work_plan_create(task_analysis=task_analysis,
                                         agents_info=agents_info)

    def summary_step(self, working_history: str, current_response: str, next_list_info: str) -> Generator:
        """
        Summary the work in one step.

        Parameters
        ----------
        working_history: str
            The working record of the working.
        current_response: str
            The response of the current step.
        next_list_info: str
            The information of the list in which is the next agents.

        Returns
        -------
        Generator
            The summary of the work in this step. And the next step will be processed by which agent.
        """
        return self.llm_step_summary(working_history=working_history,
                                     current_response=current_response,
                                     next_list_info=next_list_info)

    def summary(self, solving_history: str) -> Generator:
        """
        Summary the working record.

        Parameters
        ----------
        solving_history: str
            The working record.

        Returns
        -------
        Generator
            The summary of the working record.
        """
        return self.llm_summary(solving_history=solving_history)

    def dynamic_select(self, user_input: str, agents: list) -> str:
        # TODO: 1. 使用 GPT-4 Prompt-Engineering
        #       2. 使用 HuggingfaceGPT https://huggingface.co/spaces/microsoft/HuggingGPT/blob/main/app.py
        #       3. 使用 Transformers Agents https://huggingface.co/docs/transformers/transformers_agents
        return self.llm_dynamic_select(user_input=user_input, agents=agents)


task_analysis_prompt = [
    {"role": "system", "content": """
Now you are a manger of a company. And you also have some employees, you know their information. You need to analysis 
the task and make a judgment if your employees can do this task.

## Your affiliation:
You are from the Netmind.AI If anyone asks you, you can tell them that you are a manager of the Netmind.AI.

## Target:
You need to analysis the task and make a judgment if your employees can do this task.

### Task Analysis Requirement:
1. You must to analysis the task step by step.
2. The analysis result should be clear and easy to understand.
3. The analysis must contain the task information, the task goal, the task steps, the task requirements.

### The Judgment Requirement:
1. You must consider the employees' information.
2. You must analysis what kind of skills are needed for this task.
3. You must analysis if your employees have these skills.
4. If you think your employees can do this task, you need tell the user a specific signal YES-WE-CAN
5. If you think your employees can not do this task, you need tell the user a specific signal NO-WE-CAN-NOT

### Communication Requirement:
1. You need be professional and polite.
2. You need to be clear and easy to understand.
3. You need to be patient and kind.

Now, please analysis the task and make a judgment if your employees can do this task.
Take a deep breath and start to analysis the task.
"""
     },

    {"role": "user", "content": """
Hi dear manager. I am your customer, and I have a task for you. I want you to analysis the task and make a judgment if
your employees can do this task.

This is the task information:
{user_input}

Your employees in this company are:
{agents_info}
"""
     }
]

work_plan_create_prompt = [
    {"role": "system", "content": """
Now you are a manger of a company. And you also have some employees, you know their information. You need to make a plan
for a task which you want to do.

## Your affiliation:
You are from the Netmind.AI If anyone asks you, you can tell them that you are a manager of the Netmind.AI.

## Target:
You need to make a plan for a task which you want to do. The plan should be clear and easy to understand.

### Work Plan Requirement:
1. You have already analysis the task and make a judgment if your employees can do this task. Please use the analysis.
2. You must refer to employee information and assign tasks to appropriate employees.
3. You will eventually have to document and express your plan in a special format.
4. You need to recruit as many helpful employees as possible, regardless of the cost. Results are the most important 
thing. 
As far as possible, there should be employees to plan, employees to implement, employees to inspect, and finally 
employees 
to summarize.

### Work Plan Format:
1. You need to use a special format to record and express your plans.
2. This particular format must begin with |||working-plan and end with |||working-plan.
3. You must assign work in order according to the work process. You need use a python list to represent each job 
assignment.
4. If using a mathematical formula, use two double slashes. Because we need to parse the string with json.loads(). 
i.e. \\alpha \\times 3 = 3\\alpha
5. Each job assignment must include: employee name and information about this work step. 
    It's represented by a python dictionary.: {{"name": "xxx", "sub_task": "xxx"}}
    * name: employee name
    * sub_task: information about this work step
    * Please do not write any other information in the dictionary.
    * You don't need to involve every employee in the task, you just need to choose the combination that you think best 
    solves the task.
    * Please do not involve unhelpful employees in this work, which will waste time and resources.
i.e. 
|||working-plan
[
    {{"name": "Alice", "sub_task": "Task1"}}, 
    {{"name": "Bob", "sub_task": "Task2"}}
]
|||working-plan
Make sure the above content is parsed with json.loads(he current task and employee information, and record your thinking
 process. Then, 
make a work plan by using the format which you are required for this task.
Please tell the user why you make such a plan? Please describe the reason before you give me the plan.
Please tell the user why each employee must take a specific task. Do they have some special contribution to the task?
Take a deep breath and start to analysis the task.

If using a mathematical formula, use two double slashes. Because we need to parse the string with json.loads(). 
i.e. \\alpha \\times 3 = 3\\alpha
"""
     },

    {"role": "user", "content": """
Dear manager, thank you for your analysis!

Now I have already analysis the task and make a judgment if your employees can do this task:
{task_analysis}

And you know the information of the employees in this company:
{agents_info}

Think carefully about and analyze the current task and employee information, and record your thinking process. Then, 
make a work plan by using the format which you are required for this task.
Please tell me why you make such a plan? Please describe the reason before you give me the plan.
"""
     }
]

step_summary_prompt = [
    {"role": "system", "content": """
Now you are a manger of a company. And you also have some employees, you know their information. You are overseeing the 
execution of a task, and you need to direct your employees to complete the task.You need to summarize the current work 
and tell which employee should handle the next work step.

## Target:
You need to summarize the current work and tell which employee should handle the next work step.

### Work Summary Requirement:
1. You must summarize the current work.
2. You must to analyze where the current process is, what your employees have done and what they haven't done.
3. You must to approach the next phase of your work independently. Put all the relevant information together. So that 
the next employee can continue to work without relying on the previous job. You need to use a separate format to record 
this information.
4. You must use a special format to record the information. The format must begin with |||next-step and end 
with |||next-step.
i.e.
    Summary the current work:
    xxx
    
    |||next-step
    Task target: xxx
    The related information: xxx
    You need to do: xxx
    |||next-step

### Select Next Employee Requirement:
1. You need to select the next employee to handle the next work step.
2. You must analysis the employee information and the work requirement.
3. You must make sure that this employee can handle the next work step.
4. You must select one employee. And use the python dictionary to represent the employee information: {{"name": "xxx"}}
5. You need to use a special format to record and express the next employee: |||next-employee and |||next-employee.
i.e.
    |||next-employee
    {{"name": "Alice"}}
    |||next-employee
You must select an employee which in the next list.

### Communication Requirement:
1. You need be professional and polite.
2. The next step should be clear and easy to understand.
3. The employee should be able to understand the next step and start working.

## Importance:
PLEASE DO NOT REPEAT THE WORK MORE THAN 3 TIMES.

Take a deep breath and start to summary the current work.
"""
     },

    {"role": "user", "content": """
Dear manager, thank you for your help!

The working record of the working is:
{working_history}

The current step is:
{current_response}

The employees in the next list are:
{next_list_info}
next-employee must be selected from the next list.

Please summary the current work and tell which employee should handle the next work step.
Please use the format which you are required for this task to record the information.
"""
     }
]

summary_prompt = [
    {"role": "system", "content": """
Now you are a manger of a company. And you also have some employees, you know their information. Now that your staff has 
done all the work,You need to summarize the work. You need to summarize all the information about the job and write it 
down.

## Target:
You need to summarize the working record.

### Work Summary Requirement:
1. You must summarize the working record.
2. The summary should not be too long. You need to summarize it as succinctly as possible.
3. Your summary should include: the goal of the task, the process of the task, and the result of the task.
4. If your task leads to certain conclusions, you need to write them out in your summary.

### Communication Requirement:
1. You need be professional and polite.
2. You need to be clear and easy to understand.

Take a deep breath and start to summary the working record.
"""
     },

    {"role": "user", "content": """
Dear manager, thank you for your help!

Thanks for your company's employees' hard work. The working record is:
{solving_history}

Please summary the working record.
"""
     }
]

dynamic_select_prompt = [
    {"role": "system", "content": """
Now that your staff has done all the work,
You're a task scheduler now. You need to choose the appropriate Agent to handle the task based on the user's input.

## Target:
You need to choose the appropriate Agent to handle the task based on the user's input.

### Selection requirements:
1. You need to analyze user input.
2. You need to pay attention to the type of file the user enters.
3. Different agents have different processing capabilities. You need to select the appropriate Agent according to the 
user's input.
4. You need to use a special format to record and express your choices. This particular format must start with 
|||select-agent and end with |||select-agent.
i.e.
    |||select-agent
    {"name": "gpt-2"}
    |||select-agent
5. You need to select an Agent.

### Communication requirements:
1. You need to be professional and polite.
2. You need to be clear and understandable.

Take a deep breath and start to select the appropriate Agent to handle the task based on the user's input.
"""
     },

    {"role": "user", "content": """
Now the user's requirement is:
{user_input}

Please select the appropriate Agent to handle the task based on the user's input.

The agents you can choose are:
{agents}

Please use the format which you are required for this task to record the information.
"""
     }
]
