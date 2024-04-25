"""
===============
NotAutoPromptEngineer
===============
@file_name: not_auto_prompt_engineer.py
@author: Tianlei Shi
@date: 2024-04-24

"""

import argparse

from xyz.utils.llm.openai_client import OpenAIClient

from example.not_auto_company.prompt_generator.agents.gpt_prompt_engineer import GPTPromptEngineer
from example.not_auto_company.prompt_generator.agents.prompts_sorter import PromptsSorter


def use_cases(arg_value):
    use_case_args = []
    for case in arg_value.split('|;|'):
        use_case = {'prompt': case}
        use_case_args.append(use_case)
    return use_case_args


def set_args():
    parser = argparse.ArgumentParser(description="Prompt Engineer")
    parser.add_argument("--description", type=str, required=True,
                        help="The description of prompt functionality. Please use quotation marks if there are spaces.")
    parser.add_argument("--use_cases", type=use_cases, required=True,
                        help="Use cases of the prompt. Please use quotation marks if there are spaces; and for multiple use cases, please split by `|;|`.")
    parser.add_argument("--prompt", type=int, default=5,
                        help="The number of prompts needs to be generated.")
    parser.add_argument("--max_token", type=int, default=60,
                        help="The question which need help.")
    parser.add_argument("--k", type=int, default=32,
                        help="Constant factor that determines how much ratings change.")

    return parser.parse_args()


if __name__ == "__main__":
    args = set_args()
    
    llm_generator_client = OpenAIClient(model="gpt-4-turbo", temperature=0.9, n=args.prompt)
    llm_generation_client = OpenAIClient(model="gpt-4-turbo", max_tokens=args.max_token, temperature=0.8)
    llm_score_client = OpenAIClient(model="gpt-4-turbo", max_tokens=1, temperature=0.5, logit_bias={
            '32': 100,  # 'A' token
            '33': 100,  # 'B' token
        })

    prompt_generator = GPTPromptEngineer(llm_generator_client)
    prompt_sorter = PromptsSorter(llm_generation_client, llm_score_client, args.k)
    

    prompts = prompt_generator(test_cases=args.use_cases, description=args.description)
    print(prompts)

    comparison = prompt_sorter(test_cases=args.use_cases, description=args.description, prompts=prompts)
    print(comparison)

    print("\nWinner: \n")
    print(comparison._rows[0][0])
