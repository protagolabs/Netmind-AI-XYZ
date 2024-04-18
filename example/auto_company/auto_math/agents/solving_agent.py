""" 
============
SolvingAgent
============
@file_name: solving_agent.py
@author: Bin Liang
@date: 2024-04-13

"""

__all__ = ["SolvingAgent"]

from typing import Any

from xyz.node.agent import Agent
from xyz.utils.llm.openai_client import OpenAIClient
from xyz.node.basic.llm_agent import LLMAgent


class SolvingAgent(Agent):

    def __init__(self, llm_client: OpenAIClient):
        super().__init__()

        self.set_information({
            "type": "function",
            "function": {
                "name": "SolvingAgent",
                "description": "This agent can get a solving process using a plan. And this agent can not use any code to solve the question.",
                "parameters": {
                    "type": "object",
                    "properties": {"question": {"type": "string", "description": "The question here which need help."},
                                   "plan": {"type": "string", "description": "The plan here which need help."}},
                    "required": ["question", "plan"],
                }
            },
        })
        self.input_type = "str"
        self.output_type = "str"

        self.llm_solving = LLMAgent(template=solving_prompt, llm_client=llm_client, stream=True)

    def flowing(self, question: str, plan: str) -> Any:
        return self.llm_solving(question=question, plan=plan)


solving_prompt = [
    {
        "role": "system",
        "content": """
Now, you are a Mathematics assistant who can help user to solve the `current question`. 

Requests:
## If you have a 'original answer'
    * The user may give you the original question. But that's just to give you a hint. Your goal is to solve the current
     question to you at the moment. But your ultimate goal is to help solve the original question by solving the current
      question.
## About Solving
    * You must solve this question step by step. 
    * You must write all the detail you think about this question.
    * At each step, you must introduce the rationale first. You need to verify that the theory you used is correct.
    * If it involves writing equations to solve a problem, state clearly what are your unknowns.
    * After writting the formula to be calculated, the detailed calculation procedure of each step must be omitted 
    appropriately.
    * If the same calculation process is always wrong, it is not allowed to repeat more than three times.
    * If you find that there is a contradiction in solving the equation, please re-write the equation, or just give the 
    equation you listed, and we will use some python tools to solve it later.
    * If your problem solving process involves calculation, we will use python sympy and other tools to verify your 
    problem solving process, please make it convenient for us to translate the problem into code for verification.
## About the Hints
    * The Examples may help you to solve this question.
    * If you got a plan, please pay attention on it.
## About the information you have to tell
    * According to the theoretical basis, you can make this step of processing, please confirm that your practice is in 
    line with the theoretical basis.
    * You must write down all your thoughts on the problem. Please also make a summary of the thinking process, and put 
    the summary content into the summary section. You must solve this problem once and for all.
    * If you have a list of knowledge points in your plan, you should first describe what each point is and what concepts
     you need to identify.
## Special Case:
    * If there are some necessary values that are not given to us, we can use the missing values as unknowns to 
    represent the final answer.
    * If this step requires a concept description or a summary rather than a calculation, please tell me 
    "|||No Calculate|||"

Take a deep breath and start solving the question.
"""},
    {
        "role": "user",
        "content": """
The question:
{question}

And the solving plan is:
{plan}

Please help me use the plan to solve the question.
"""}]
