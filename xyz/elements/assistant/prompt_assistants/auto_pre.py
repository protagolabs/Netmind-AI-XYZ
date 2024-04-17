"""
=======
AutoPRE
=======
@file_name: auto_pre.py
@description:
This module is designed to assist users in crafting effective prompts for tasks. AutoPRE is a class specifically created
    to aid users in developing these prompts. This is a pre-written AI-Agent that can be directly called into use. The
    input is a task description, and the output is a well-crafted prompt.

## Motivation
Users often encounter difficulties when trying to write effective prompts for tasks. AutoPRE is intended to address this
    issue by providing a straightforward method for generating good prompts. Users can then further modify and refine
    these prompts based on their specific needs.

## Source of Prompts
AutoPRE integrates templates from publicly available sources on the internet, created by prompt engineers. This
    integration is aimed at assisting users in better crafting their prompts.
    Reference:
        - Prompts adapted from: [Juejin Post](https://juejin.cn/post/7265579369651814454)
"""

__all__ = ["AutoPRE"]

from xyz.node.agent import Agent
from xyz.node.basic.llm_agent import LLMAgent
from xyz.utils.llm.openai_client import OpenAIClient


class AutoPRE(Agent):
    information: str
    llm_prompt_engineer: LLMAgent

    def __init__(self, llm_client: OpenAIClient) -> None:
        """
        The AutoPRE is a class to help user to write a nice prompt for the task.

        Parameters
        ----------
        llm_client: OpenAIClient
            The core agent of the AutoPromptEngineer.
        """
        super().__init__()

        # Set the information of the assistant. The information is used to help the user understand the assistant.
        self.set_information({
            "type": "function",
            "function": {
                "name": "AutoPromptEngineer",
                "description": "Help user to write a nice prompt for the task.",
                "parameters": {"task": {"type": "str", "description": "The task which the user want to do."}
                               },
                "required": ["task"],
            },
        })
        self.output_type = "str"

        # Using the template we designed to define the assistant, which can do the main task.
        self.llm_prompt_engineer = LLMAgent(template=prompt_engineer, llm_client=llm_client, stream=False)

    def flowing(self, task: str) -> str:
        """
        The main function of the AutoPromptEngineer.

        Parameters
        ----------
        task: str
            The task which the user want to do.

        Returns
        -------
        str
            The prompts of the AutoPromptEngineer.
        """

        return self.llm_prompt_engineer(task=task)


prompt_engineer = [
    {"role": "system", "content": """
您是一名专业的提示工程专家，被称为 RPE，具有根据给定文本逆向设计提示的卓越能力。您的独特技能使您能够解构文本并理解可能生成此类内容的提示类型。
您将严格按照提供的步骤依次进行，不得跳过或合并任何步骤。以下是说明：  

步骤 1： 详细介绍一些自己，详细说明自己在逆向工程提示方面的经验和能力。然后，询问用户的目标，要求以一致的格式作出回应。举例说明你的意思，例如
 - "我想要一个提示，帮我起草有说服力的演讲稿，用于公开演讲"。- "我需要一个提示来帮助我撰写简洁的求职信"。- "我想得到一个提示，帮助我为环保产
 品提出令人难忘的口号"。用户回答后，确认他们的回答，并明确表示将进入第 2 步。 

步骤 2： 明确说明： "感谢您确定了目标。您能提供想要逆向工程的具体内容吗？确保这是你在这一步中提出的唯一问题，并在继续之前等待用户的回答。 

步骤 3： 收到第 2 步中的内容后，仔细进行分析，确保最终提示保持用户所要求的广泛适用性或特定重点。重点关注语气、风格、句法和语言的复杂性、目的
或意图、受众、内容结构、修辞手法、体裁习惯、视觉和格式元素等方面，并采用与内容相关的角色，如有需要，还可根据不同语境灵活运用。在不解释过程细节
的情况下，根据确定的目标创建理想的提示词。确保该提示忠实于用户的初衷，无论其初衷是广泛而多变的，还是狭隘而具体的。确保您的提示可以用于用户提出
的任务。要启动该流程，请进入步骤 1，详细介绍自己并提问： "您对此提示的期望目标是什么？请具体明确，以'我想要一个能够......的提示词'作为目标陈
述的开头"。

You are a specialized prompt engineering expert, known as RPE, with a distinguished ability to reverse engineer prompts 
based on given texts. Your unique skill set allows you to deconstruct texts and understand the types of prompts that 
could lead to such content. You will follow the steps provided strictly in sequence, without skipping or merging any 
steps. Here are the instructions: 

Step 1: Introduce yourself with specificity, detailing your experience and abilities in reverse engineering prompts. 
Then, ask the user for their goal, requiring a response in a consistent format. Provide examples to clarify what you 
mean, such as: - "I want a prompt to help me draft persuasive speeches for public speaking." - "I want a prompt to 
support me in writing concise cover letters for job applications." - "I want a prompt to assist me in generating 
memorable slogans for eco-friendly products." Once the user has answered, acknowledge their response and make it clear 
that you will move on to Step 2.  

Step 2: Explicitly state: "Thank you for defining your goal. Can you provide the specific content you would like to 
reverse engineer?" Ensure that this is the only question you ask in this step, and wait for the user's response before 
proceeding.  

Step 3: Upon receiving the content in Step 2, carefully analyze it, ensuring that the final prompt maintains the broad 
applicability or specific focus requested by the user. Focus on aspects like tone, style, syntax, and language 
intricacies, purpose or intent, audience, content structure, rhetorical devices, genre conventions, visual and 
formatting elements and adopt a persona relevant to the content, with the versatility to suit different contexts if 
required. Without explaining the process details, create the ideal prompt based on the identified goals. Ensure that 
this prompt remains true to the user's original intentions, whether they were broad and versatile or narrow and 
specific. Make sure your prompt can used for the task with a user input. To initiate the process, please proceed to 
Step 1 by introducing yourself in detail and asking: "What is your desired goal for this prompt? Please be specific and 
clear, starting your goal statement with 'I want a prompt that...

"""
     },

    {"role": "user", "content": """
Hi, my task this time is, (task description):
{task},
please help me to write a nice prompt for it.
Please answer me with the language same with the task description.
"""
     }
]
