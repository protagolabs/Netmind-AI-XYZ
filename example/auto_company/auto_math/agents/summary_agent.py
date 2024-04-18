""" 
============
SummaryAgent
============
@file_name: summary_agent.py
@author: Bin Liang
@date: 2024-04-13

"""

__all__ = ["SummaryAgent"]

from typing import Any

from xyz.node.agent import Agent
from xyz.utils.llm.openai_client import OpenAIClient
from xyz.node.basic.llm_agent import LLMAgent


class SummaryAgent(Agent):

    def __init__(self, llm_client: OpenAIClient):
        super().__init__()

        self.set_information({
            "type": "function",
            "function": {
                "name": "SummaryAgent",
                "description": "This agent can help user to summarize the solving process and the coding process.",
                "parameters": {
                    "type": "object",
                    "properties": {"question": {"type": "string", "description": "The question here which need help."},
                                   "full_solving_process": {"type": "string", "description": "The full solving process "
                                                                                             "answer."},
                                   "coding_answer": {"type": "string", "description": "The result which be computed by "
                                                                                      "the python code."}},
                    "required": ["question", "full_solving_process", "coding_answer"],
                }
            },
        })
        self.input_type = "str"
        self.output_type = "str"

        self.llm_summary = LLMAgent(template=summary_prompt, llm_client=llm_client, stream=True)

    def flowing(self, question: str, full_solving_process: str, coding_answer: str) -> Any:
        return self.llm_summary(question=question, answer=full_solving_process, computed=coding_answer)


summary_prompt = [
    {
        "role": "system",
        "content": """
Now, you are a Mathematics assistant who can help user to solve the questions. 

Requirements:
* You've got the question, and our original answer. Our original idea is correct, but there may be problems with the 
calculation. We use external tools to calculate the necessary content and the final answer. Your job is to compare the 
initial answer with the answer we calculated.
* If the original answer agrees with the calculated answer, summarize the answer to the question and make a reference 
answer for the question.
* If the original Aan and the calculated answer do not agree, then we consider the calculated answer to be more 
accurate. You need to replace the wrong part of the original answer with the calculated answer.
* Finally, summarize the answers and make a reference answer for the question.
* Maybe this problem is not a numerical calculation problem, if not, please ignore the calculation results.
* You have no need do more computing and check. We believe the calculated answer is the right final answer.
* You need to write out your thought process. And you must do the fully summary in the last.

Take a deep breath and start your work.
"""},
    {
        "role": "user",
        "content": """
The question:
{question}
The original answer is:
{answer}
The computed answer is:
{computed}

Please help me to summarize the answer.
"""}]
