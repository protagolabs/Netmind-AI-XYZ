"""
=============
PromptsSorter
=============
@file_name: prompts_sorter.py
@author: Tianlei Shi
@date: 2024-4-24
"""


__all__ = ["PromptsSorter"]

from prettytable import PrettyTable
import openai
from tqdm import tqdm
import itertools
from tenacity import retry, stop_after_attempt, wait_exponential

from xyz.node.agent import Agent
from xyz.node.basic.llm_agent import LLMAgent
from xyz.utils.llm.openai_client import OpenAIClient


class PromptsSorter(Agent):
    information: str
    llm_prompt_engineer: LLMAgent
    
    N_RETRIES = 3  # number of times to retry a call to the ranking model if it fails

    def __init__(self, generation_agent: OpenAIClient, score_agent: OpenAIClient, k: int) -> None:

        super().__init__()

        # Set the information of the assistant. The information is used to help the user understand the assistant.
        self.set_information({
            "type": "function",
            "function": {
                "name": "PromptsSorter",
                "description": "Sort prompts to find an optimal prompt for a given task.",
                "parameters": {
                    "type": "object",
                    "properties": {"test_cases": {"type": "list", "description": "Some examples of the task."},
                                   "description": {"type": "string", "description": "Description of the task which the user want to do."},
                                   "prompts": {"type": "list", "description": "Prompts that need to be compared."}
                                   },
                "required": ["test_cases", "description", "prompts"],
                }
            },
        })
        self.input_type = "str"
        self.output_type = "str"

        # Using the template we designed to define the assistant, which can do the main task.
        self.llm_generation_agent = LLMAgent(template=generation_prompt, llm_client=generation_agent, stream=False)
        self.llm_score_agent = LLMAgent(template=ranking_prompt, llm_client=score_agent, stream=False)
        
        self.K = k  # K is a constant factor that determines how much ratings change
    
    
    def expected_score(self, r1, r2):
        # Calculate the expected score (win rate) between two prompts based on their current ELO ratings
        # The one has higher ELO rating will have higher score
        
        # latex format: \frac{1}{1 + 10^{\frac{r2 - r1}{400}}}
        return 1 / (1 + 10**((r2 - r1) / 400))
    

    def update_elo(self, r1, r2, score1):
        # update their ELO ratings based on comparison result and their expected score (win rate)
        
        # r = r + K × (S − E)
        # where r is the current ELO rating
        # K is a constant factor (sensitivity of the score adjustment). The larger the K, the greater the impact of a single comparison result on the rating
        # S is the actual score (win=1, loss=0, draw=0.5)
        # E is the expected score
        
        e1 = self.expected_score(r1, r2)
        e2 = self.expected_score(r2, r1)
        return r1 + self.K * (score1 - e1), r2 + self.K * ((1 - score1) - e2)
    
    
    # Get Score - retry up to N_RETRIES times, waiting exponentially between retries.
    @retry(stop=stop_after_attempt(N_RETRIES), wait=wait_exponential(multiplier=1, min=4, max=70))
    def get_score(self, description, test_case, pos1, pos2):
        # Compare results of the two prompts, and find the winner or draw
        return self.llm_score_agent(description=description.strip(), test_case=test_case['prompt'], pos1=pos1, pos2=pos2)
    
    
    @retry(stop=stop_after_attempt(N_RETRIES), wait=wait_exponential(multiplier=1, min=4, max=70))
    def get_generation(self, prompt, test_case):
        # Generate results for each test case by using a specific prompt
        return self.llm_generation_agent(prompt=prompt, test_case=test_case['prompt'])
    
    
    def test_candidate_prompts(self, test_cases, description, prompts):
        """
        This method compares prompts pairly to get the optimal one. 
        Each prompt have an initial ELO rating, and then increase its rating if it wins, or decrease otherwise.

        Parameters
        ----------
        test_cases: dict
            Test cases of the given task.
        description: str
            Description of the given task.
        prompts: list
            A list of prompts of the given task that wait to be compared.
        """
        
        # Initialize each prompt with an ELO rating of 1200
        prompt_ratings = {prompt: 1200 for prompt in prompts}

        # Calculate total rounds for progress bar
        total_rounds = len(test_cases) * len(prompts) * (len(prompts) - 1) // 2

        # Initialize progress bar
        pbar = tqdm(total=total_rounds, ncols=70)

        # For each pair of prompts
        for prompt1, prompt2 in itertools.combinations(prompts, 2):
            # For each test case
            for test_case in test_cases:
                # Update progress bar
                pbar.update()

                # Generate outputs for each prompt
                generation1 = self.get_generation(prompt1, test_case)
                generation2 = self.get_generation(prompt2, test_case)

                # Rank the outputs
                score1 = self.get_score(description, test_case, generation1, generation2)
                score2 = self.get_score(description, test_case, generation2, generation1)

                # Convert scores to numeric values
                score1 = 1 if score1 == 'A' else 0 if score1 == 'B' else 0.5
                score2 = 1 if score2 == 'B' else 0 if score2 == 'A' else 0.5

                # Average the scores
                score = (score1 + score2) / 2

                # Update ELO ratings
                r1, r2 = prompt_ratings[prompt1], prompt_ratings[prompt2]
                r1, r2 = self.update_elo(r1, r2, score)
                prompt_ratings[prompt1], prompt_ratings[prompt2] = r1, r2

                # Print the winner of this round
                if score > 0.5:
                    print(f"Winner: {prompt1}")
                elif score < 0.5:
                    print(f"Winner: {prompt2}")
                else:
                    print("Draw")

        # Close progress bar
        pbar.close()

        return prompt_ratings
    

    def flowing(self, test_cases, description, prompts) -> str:
        
        prompt_ratings = self.test_candidate_prompts(test_cases, description, prompts)

        # Print the final ELO ratingsz
        table = PrettyTable()
        table.field_names = ["Prompt", "Rating"]
        for prompt, rating in sorted(prompt_ratings.items(), key=lambda item: item[1], reverse=True):
            table.add_row([prompt, rating])

        return table



ranking_prompt = [
    {"role": "system", "content": """Your job is to rank the quality of two outputs generated by different prompts. The prompts are used to generate a response for a given task.

You will be provided with the task description, the test prompt, and two generations - one for each system prompt.

Rank the generations in order of quality. If Generation A is better, respond with 'A'. If Generation B is better, respond with 'B'.

Remember, to be considered 'better', a generation must not just be good, it must be noticeably superior to the other.

Also, keep in mind that you are a very harsh critic. Only rank a generation as better if it truly impresses you more than the other.

Respond with your ranking, and nothing else. Be fair and unbiased in your judgement.
"""
    },
    {"role": "user", "content": """Task: {description}
        Prompt: {test_case}
        Generation A: {pos1}
        Generation B: {pos2}"""}
]


generation_prompt = [
    {"role": "system", "content": """{prompt}"""},
    {"role": "user", "content": "{test_case}"}
]