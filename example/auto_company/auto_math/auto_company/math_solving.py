"""
===============
AutoMathCompany
===============
@file_name: auto_math_company.py
@author: Bin Liang
@date: 2024-04-13

"""

import argparse

from xyz.graph.auto_company import AutoCompany
from xyz.utils.llm.openai_client import OpenAIClient

from example.auto_company.auto_math.agents.solving_agent import SolvingAgent

def main(api_key, question):

    llm_client = OpenAIClient(api_key=api_key, model="gpt-4-turbo")

    solving_agent = SolvingAgent(llm_client)
    
    return solving_agent(user_input=question)
