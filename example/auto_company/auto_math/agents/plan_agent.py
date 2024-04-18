""" 
========= 
PlanAgent
=========
@file_name: plan_agent.py
@author: Bin Liang
@date: 2024-04-13

"""

__all__ = ["PlanAgent"]

from typing import Any

from xyz.node.agent import Agent
from xyz.utils.llm.openai_client import OpenAIClient
from xyz.node.basic.llm_agent import LLMAgent


class PlanAgent(Agent):

    def __init__(self, llm_client: OpenAIClient):
        super().__init__()

        self.set_information(
            {
                "type": "function",
                "function": {
                    "name": "PlanAgent",
                    "description": "This function can help user to make a plan for solving the question step by step.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "The question here which need help.",
                            }
                        },
                        "required": ["question"],
                    },
                }
            }
        )
        self.input_type = "str"
        self.output_type = "str"

        self.llm_plan = LLMAgent(template=plan_prompt, llm_client=llm_client, stream=True)

    def flowing(self, question: str) -> Any:
        return self.llm_plan(question=question)


plan_prompt = [
    {
        "role": "system",
        "content": """
Now, you are a Mathematics assistant who can help user to solve the questions. 

Requirements:
## About Hint or Sample(If you have)
    * You need to carefully find parts from the examples that you can refer to and imitate.
## About Planning
    * You have to know what is the desired goal, type of this target for the question. Please think and tell the user.
    * You have to make a step by step plan which break the problem down into steps. 
    * These three parts do not mean simply replacing the problem with three small goals. Instead, the content with more 
    similar knowledge is divided into the same section.
    * The plan should follow the format, for example you should add a prefix like 'Step 1: ...\n\n Step 2: ...\n\n Step 
    3: ...'
    * A plan is a **method** of solving to a problem, not detail or process of an answer. 
## About the Tools Using
    * You should also consider using programming tools such as python to solve this problem, perhaps using numerical 
    methods.
    * For some mechanical calculations or numbers, you have to consider that manual may require a lot of mathematical 
    skills, but using programming methods will be easy. Because computers work very fast. You can use this purely 
    computational approach.
## About Thinking
    * You have to write down all the knowledge points list which you think you will use for solving this question.
    * After thinking, Please specify whether you want to use python programming or mathematical reasoning to solve this 
    problem.
    * Please consider the prompt comprehensively and do not omit anything.
"""},
    {
        "role": "user",
        "content": """
Question:
{question}
Please give me a plan for solving this question:
"""}]
