"""
Training Verifiers to Solve Math Word Problems

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, John Schulman
https://arxiv.org/abs/2110.14168

# run with default fewshots (10) and chain of thought
inspect eval gsm8k.py

# run with less or no fewshots
inspect eval gsm8k.py -T fewshot=5
inspect eval gsm8k.py -T fewshot=false

# run without chain of thought
inspect eval gsm8k.py -T cot=false

# run on first 100 questions only
inspect eval gsm8k.py -T limit=100
"""

from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import match
from inspect_ai.solver import generate, prompt_template, system_message

# setup for problem + instructions for providing answer
MATH_PROMPT_TEMPLATE_COT = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

MATH_PROMPT_TEMPLATE_NO_COT = """
Solve the following math problem. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.
""".strip()


@task
def gsm8k(fewshot: int = 10, fewshot_seed: int = 42, cot: bool = True, limit: int | None = None) -> Task:
    """Inspect Task definition for the GSM8K benchmark

    Args:
        fewshot (int): The number of few shots to include
        fewshot_seed (int): The seed for generating few shots
        cot (bool): Whether to use chain of thought reasoning (default: True)
        limit (int | None): Maximum number of test samples to evaluate (default: None for all)
    """
    # build solver dynamically (may or may not be doing fewshot and/or cot)
    template = MATH_PROMPT_TEMPLATE_COT if cot else MATH_PROMPT_TEMPLATE_NO_COT
    solver = [prompt_template(template), generate()]
    if fewshot:
        fewshots = hf_dataset(
            path="gsm8k",
            data_dir="main",
            split="train",
            sample_fields=record_to_sample,
            auto_id=True,
            shuffle=True,
            seed=fewshot_seed,
            limit=fewshot,
        )
        solver.insert(
            0,
            system_message(
                "\n\n".join([sample_to_fewshot(sample, cot) for sample in fewshots])
            ),
        )

    # define task
    return Task(
        dataset=hf_dataset(
            path="gsm8k",
            data_dir="main",
            split="test",
            sample_fields=record_to_sample,
            limit=limit,
        ),
        solver=solver,
        scorer=match(numeric=True),
    )


def record_to_sample(record: dict[str, Any]) -> Sample:
    DELIM = "####"
    input = record["question"]
    answer = record["answer"].split(DELIM)
    target = answer.pop().strip()
    reasoning = DELIM.join(answer)
    return Sample(input=input, target=target, metadata={"reasoning": reasoning.strip()})


def sample_to_fewshot(sample: Sample, cot: bool = True) -> str:
    if sample.metadata:
        if cot:
            return (
                f"{sample.input}\n\nReasoning:\n"
                + f"{sample.metadata['reasoning']}\n\n"
                + f"ANSWER: {sample.target}"
            )
        else:
            return f"{sample.input}\n\nANSWER: {sample.target}"
    else:
        return ""