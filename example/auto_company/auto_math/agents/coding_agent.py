""" 
===========
CodingAgent
===========
@file_name: coding_agent.py
@date: 2023-8-17
@author: Bin Liang
"""

__all__ = ["CodingAgent"]

import sys
from io import StringIO
import io
import contextlib
import traceback

from xyz.node.agent import Agent
from xyz.utils.llm.openai_client import OpenAIClient
from xyz.node.basic.llm_agent import LLMAgent


def extract_code_blocks(text):
    import re

    pattern = r'```python(.*?)```'
    result = re.findall(pattern, text, re.DOTALL)
    result = "\n".join(result)
    return result


class CodingAgent(Agent):

    def __init__(self, llm_client: OpenAIClient):
        super().__init__()

        self.set_information({
            "type": "function",
            "function": {
                "name": "CodingAgent",
                "description": "This agent can write a python code to compute the math part by following the solving "
                               "process.",
                "parameters": {
                    "type": "object",
                    "properties": {"question": {"type": "string", "description": "The question here which need help."},
                                   "full_solving_process": {"type": "string", "description": "The full solving process "
                                                                                             "answer."}},
                    "required": ["question", "full_solving_process"],
                }
            },
        })
        self.input_type = "str"
        self.output_type = "str"

        self.repeat_time = 0

        self.coding_agent = LLMAgent(template=coding_prompt, llm_client=llm_client, stream=True)
        self.debug_agent = LLMAgent(template=debug_prompt, llm_client=llm_client, stream=False)

    def flowing(self, question: str, full_solving_process: str):

        python_code_generate = self.coding_agent(question=question, input=full_solving_process)

        python_code_rough = ""
        for code in python_code_generate:
            python_code_rough += code
            yield code

        yield f"\nRunning this code...Please wait...\n"

        right_answer = self.run_code(python_code_rough)

        yield f"The running_code is over...\n{right_answer}"

        new_answer = (f"The python code is here(you can know the process from the code): \n" + python_code_rough +
                      f"\nAnd the computed answer is: \n" + right_answer)

        yield new_answer

    def help_runcode(self, sample):

        if "```python\n" in sample:
            sample = sample.replace("```python\n", '')
        if "```python" in sample:
            sample = sample.replace("```python", '')
        if "python" in sample:
            sample = sample.replace("python", '')
        if "```\n" in sample:
            sample = sample.replace("```\n", '')
        if "\n```" in sample:
            sample = sample.replace("\n```", '')
        if "```" in sample:
            sample = sample.replace("```", '')

        if "Python code: " in sample:
            sample = sample.replace("Python code: ", '')

        sample = sample.strip()
        with self.stdoutIO() as s:
            try:
                exec(sample)
                return s.getvalue()
            except:
                return ""

    def run_code(self, sample):

        if "```" in sample:
            sample = extract_code_blocks(sample)
        else:
            pass

        sample = sample.strip()
        output = io.StringIO()
        error_message = None

        try:
            with contextlib.redirect_stdout(output):
                exec(sample, globals())
        except Exception:
            error_message = traceback.format_exc()

        print_output = output.getvalue()

        if error_message:
            self.repeat_time += 1
            if self.repeat_time >= 4:
                return "The code is error."

            new_sample = self.debug_agent(code=sample, error=error_message)
            new_code = new_sample.split("|||Code:")[-1]
            new_result = self.run_code(new_code)
            return new_result
        else:
            return print_output

    @contextlib.contextmanager
    def stdoutIO(self, stdout=None):
        old = sys.stdout
        if stdout is None:
            stdout = StringIO()
        sys.stdout = stdout
        yield stdout
        sys.stdout = old


coding_prompt = [
    {
        "role": "system",
        "content": """ 
# Guidance

Previous is a question and its answer, the solving idea is right but the calculation could be wrong.

Do not answer the question. Instead, your task is to write some numeric calculation and comparisons relevant to 
answering the question.

After the question write a python code block with up to four sections containing content relevant to answering the 
question.

You should not write anything other than code itself. You should not comment the code or explain the code. 

In the "Definitions" section define a label for each number in the
original question like `car_count` or `speed_of_car_in_km_per_hour`.
* Every label name should include the unit of measure if known.
* This section should be valid Python and can include valid Python
  single-dimensional arrays.
* Do not use or create multi-dimensional arrays.
* Give each label a unit of measure in a comment after each definition.
* Document the meaning of each definition in the comment.
* If the unit of measure is unknown use "unknown".
* Omit this section if there are no numbers in the question.
* If there are unknowns, to set 'symbols', you must set all symbols in this section.
* Please import the package which you will ues for computing. 
* You're not just verifying the user's answers. Do it yourself in a few ways.

In the "Calculations" section define additional relevant labels using Python or numpy formulae or sympy. 
* Please use the newest version python package. * Define each label using a formula, referencing previously defined 
labels. 
* Avoid new assumptions in this section, if you make an assumption document it. 
* Every label name should include the unit of measure if known. 
* Do NOT include the calculated values for these labels. 
* Give each label a unit of measure in a comment after each definition. 
* Document the meaning of each definition in the comment. 
* If the unit of measure is unknown use "unknown". 
* This section should be valid Python using regular Python or numpy. 
* Omit this section if there are no additional labels relevant to the answer. 
* If it involves calculus operations, such as derivations, integrals, algebraic operations, you can use the python sympy 
package. 
* If it involves solving equations and seeking the best value, you should also choose the right package to use, 
recommend the use of sympy. 
* For overly complex calculations, you can also use some numerical methods to solve the problem. 
* You must completely repeat the calculation process in the answer! 
* Do not plot any figure! 
* Do not ignore any process in the process of solving the problem, which must be recalculated in code. 
* The content you generate should be ready to run as a python script without any special processing. 
* You should have an expectation of how long the code will take to run. It should not run for more than 1 minute.

In the "Comparisons" section define additional labels using Python or numpy
formulae by comparing labels using comparison operators and functions and
evaluating to single boolean values.
* Do NOT include the calculated true/false values for these labels.
* This section should be valid Python using regular Python or numpy.
* Document the meaning of each definition in the comment.
* Omit this section if there are no comparisons relevant to the answer.

In the "Evaluations" section print the values of pre-defined labels using python.
* This section should be valid Python using regular Python or numpy.
* The values of pre-defined labels should be attached to their label names.
* All pre-defined labels should be printed.
* Please do not print too many intermediate processes. Users care more about the end result.

The python code is:
"""},
    {
        "role": "user",
        "content": """ 
The question is:
{question}
The answer we try is:
{input}
"""}
]

debug_prompt = [

    {"role": "system",
     "content": """ 
You are an experienced numerical computing python engineer.

Your job is to help people fix bugs and make sure the program works.

Requirements: * Please do not add additional third packages * Please first think about why such a bug occurs * Please 
write down your thought process * Once you start writing code, don't say anything else. * Please pay attention to the 
format, so that users can clearly find the modified Code, please write the special mark '|||Code:\n', and then follow 
your modified code. The content after this special tag should be python code that can be run. * '|||Code:\n' After 
that, only code"""},

    {"role": "user",
     "content": """
Original code:
{code}
Error message:
{error}
"""}]
