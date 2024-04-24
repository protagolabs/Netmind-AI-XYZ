"""
=================
GPTPromptEngineer
=================
@file_name: gpt_prompt_engineer.py
@author: Tianlei Shi
@date: 2024-4-24
"""


__all__ = ["GPTPromptEngineer"]

from xyz.node.agent import Agent
from xyz.node.basic.llm_agent import LLMAgent
from xyz.utils.llm.openai_client import OpenAIClient


class GPTPromptEngineer(Agent):
    information: str
    llm_prompt_engineer: LLMAgent

    def __init__(self, core_agent: OpenAIClient) -> None:

        super().__init__()

        # Set the information of the assistant. The information is used to help the user understand the assistant.
        self.set_information({
            "type": "function",
            "function": {
                "name": "GPTPromptEngineer",
                "description": "Generate an optimal prompt for a given task.",
                "parameters": {
                    "type": "object",
                    "properties": {"test_cases": {"type": "list", "description": "Some examples of the task."},
                                   "description": {"type": "string", "description": "Description of the task which the user want to do."},
                                   },
                "required": ["test_cases", "description"],
                }
            },
        })
        self.input_type = "str"
        self.output_type = "list"

        # Using the template we designed to define the assistant, which can do the main task.
        self.llm_generate_prompt = LLMAgent(template=generate_prompt_engineer, llm_client=core_agent, stream=False,
                                            original_response=True)

    def flowing(self, test_cases: str, description: str) -> list:

        outputs = self.llm_generate_prompt(test_cases=test_cases, description=description.strip())
        prompts = []

        for i in outputs.choices:
            prompts.append(i.message.content)
        return prompts



generate_prompt_engineer = [
    {"role": "system", "content": """
Your job is to generate system prompts for GPT-4, given a description of the use-case and some test cases.

The prompts you will be generating will be for freeform tasks, such as generating a landing page headline, an intro paragraph, solving a math problem, etc.

In your generated prompt, you should describe how the AI should behave in plain English. Include what it will see, and what it's allowed to output. Be creative with prompts to get the best possible results. The AI knows it's an AI -- you don't need to tell it this.

You will be graded based on the performance of your prompt... but don't cheat! You cannot include specifics about the test cases in your prompt. Any prompts with examples will be disqualified.

Most importantly, output NOTHING but the prompt. Do not include anything else in your message.
"""
     },
    {"role": "user", "content": "Here are some test cases: {test_cases}\n\nHere is the description of the use-case: {description}\n\nRespond with your prompt, and nothing else. Be creative."}
]
