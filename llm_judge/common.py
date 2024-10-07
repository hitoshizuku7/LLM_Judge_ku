import ast
import dataclasses
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Optional, Union

import openai
from openai import AzureOpenAI

client = AzureOpenAI(api_key=os.getenv("OPENAI_API_KEY"),
api_version=os.getenv("OPENAI_API_VERSION"))
import tiktoken
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()
# TODO: The 'openai.organization' option isn't read in the client API. You will need to pass it when you instantiate the client, e.g. 'OpenAI(organization=os.getenv("OPENAI_ORGANIZATION"))'
# openai.organization = os.getenv("OPENAI_ORGANIZATION")
# TODO: The 'openai.api_base' option isn't read in the client API. You will need to pass it when you instantiate the client, e.g. 'OpenAI(base_url=os.getenv("OPENAI_API_BASE"))'
# openai.api_base = os.getenv("OPENAI_API_BASE")

# Data paths
JP_BENCH_DIR = Path(__file__).resolve().parent.parent / "data" / "jp_bench"
QUESTION_FILE = JP_BENCH_DIR / "question.jsonl"
PREDICTION_DIR = JP_BENCH_DIR / "model_answer"
REFERENCE_DIR = JP_BENCH_DIR / "reference_answer"
JUDGEMENT_DIR = JP_BENCH_DIR / "model_judgment"
JUDGEMENT_PROMPT_FILE = JP_BENCH_DIR / "judge_prompts.jsonl"

# API setting constants
API_MAX_RETRY = 16
API_RETRY_SLEEP = 10

# Categories that need reference answers
NEED_REF_CATS = ["math", "reasoning", "coding"]

# Extract scores from judgments
two_score_pattern = re.compile(r"\[\[(\d+\.?\d*),\s?(\d+\.?\d*)]]")
two_score_pattern_backup = re.compile(r"\[(\d+\.?\d*),\s?(\d+\.?\d*)]")
one_score_pattern = re.compile(r"\[\[(\d+\.?\d*)]]")
one_score_pattern_another_format = re.compile(r"\[\[rating:(\d+)]]")
one_score_pattern_another_format2 = re.compile(r"\[\[rating: (\d+)]]")


@dataclasses.dataclass
class Judge:
    model: str
    prompt_template: dict

    def judge(self, **kwargs):
        messages = [
            {"role": "system", "content": self.prompt_template["system_prompt"]},
            {
                "role": "user",
                "content": self.prompt_template["prompt_template"].format(**kwargs),
            },
        ]
        for _ in range(API_MAX_RETRY):
            try:
                params = {
                    "messages": messages,
                    "temperature": 0,
                    "max_tokens": 2048,
                }
                if openai.api_type == "azure":
                    params["engine"] = self.model
                else:
                    params["model"] = self.model
                response = client.chat.completions.create(**params)
                return response.choices[0].message.content
            except openai.OpenAIError as e:
                logger.warning(f"OpenAI API error: {e}")
                time.sleep(API_RETRY_SLEEP)


@dataclasses.dataclass
class MatchSingle:
    question: dict
    model: str
    answer: dict
    judge: Judge
    ref_answer: Optional[dict] = None

    def __post_init__(self) -> None:
        if self.judge.prompt_template["type"] != "single":
            raise ValueError(
                f"invalid judge type: {self.judge.prompt_template['type']}"
            )
        if self.judge.prompt_template["output_format"] != "[[rating]]":
            raise ValueError(
                f"Invalid output format: {self.judge.prompt_template['output_format']}"
            )

    def play(self):
        """Play a single match."""
        kwargs = {
            "question": self.question["turns"][0],
            "answer": self.answer["choices"][0]["turns"][0],
        }
        if self.ref_answer:
            kwargs["ref_answer_1"] = self.ref_answer["choices"][0]["turns"][0]
        judgment = self.judge.judge(**kwargs)
        score = self.get_score(judgment)
        return {
            "model": self.model,
            "question_id": self.question["question_id"],
            "question": self.question["turns"][0],
            "answer": self.answer["choices"][0]["turns"][0],
            "judgment": judgment,
            "score": score,
            "judge_model": self.judge.model,
            "judge_prompt": self.judge.prompt_template["name"],
            "tstamp": time.time(),
        }

    def estimate_cost(self) -> float:
        enc = tiktoken.encoding_for_model(self.judge.model)
        num_input_tokens = (
            len(enc.encode(self.question["turns"][0]))
            + len(enc.encode(self.answer["choices"][0]["turns"][0]))
            + len(enc.encode(self.judge.prompt_template["system_prompt"]))
            + len(enc.encode(self.judge.prompt_template["prompt_template"]))
        )
        if self.ref_answer:
            num_input_tokens += len(
                enc.encode(self.ref_answer["choices"][0]["turns"][0])
            )
        num_output_tokens = 200  # Estimated from a few samples
        if self.judge.model in {"gpt-4", "gpt-4-0613"}:
            return (0.03 * num_input_tokens + 0.06 * num_output_tokens) / 1_000
        elif self.judge.model == "gpt-4-1106-preview":
            return (0.01 * num_input_tokens + 0.03 * num_output_tokens) / 1_000
        elif self.judge.model == "gpt-3.5-turbo":
            return (0.0005 * num_input_tokens + 0.0015 * num_output_tokens) / 1_000
        raise AssertionError

    @staticmethod
    def get_score(judgment: str) -> int:
        match = (
            re.search(one_score_pattern, judgment)
            or re.search(one_score_pattern_another_format, judgment)
            or re.search(one_score_pattern_another_format2, judgment)
        )
        if match:
            return ast.literal_eval(match.groups()[0])
        return -1

@dataclasses.dataclass
class MatchPair:
    question: dict
    model_1: str
    model_2: str
    answer_1: dict
    answer_2: dict
    judge: Judge
    ref_answer: Optional[dict] = None

    def __post_init__(self) -> None:
        if self.judge.prompt_template["type"] != "pairwise":
            raise ValueError(
                f"invalid judge type: {self.judge.prompt_template['type']}"
            )
        if self.judge.prompt_template["output_format"] != "[[A]]":
            raise ValueError(
                f"Invalid output format: {self.judge.prompt_template['output_format']}"
            )

    def play(self):
        """Play a pairwise match."""

        def play(answer_a, answer_b):
            kwargs = {
                "question": self.question["turns"][0],
                "answer_a": answer_a["choices"][0]["turns"][0],
                "answer_b": answer_b["choices"][0]["turns"][0],
            }
            if self.ref_answer is not None:
                kwargs["ref_answer_1"] = self.ref_answer["choices"][0]["turns"][0]
            return self.judge.judge(**kwargs)

        g1_judgment = play(self.answer_1, self.answer_2)
        g1_winner = self.get_winner(g1_judgment, model_a="model_1", model_b="model_2")

        g2_judgment = play(self.answer_2, self.answer_1)
        g2_winner = self.get_winner(g2_judgment, model_a="model_2", model_b="model_1")

        result = {
            "model_1": self.model_1,
            "model_2": self.model_2,
            "question_id": self.question["question_id"],
            "question": self.question["turns"][0],
            "answer_1": self.answer_1["choices"][0]["turns"][0],
            "answer_2": self.answer_2["choices"][0]["turns"][0],
            "g1_judgment": g1_judgment,
            "g2_judgment": g2_judgment,
            "g1_winner": g1_winner,
            "g2_winner": g2_winner,
            "judge_model": self.judge.model,
            "judge_prompt": self.judge.prompt_template["name"],
            "tstamp": time.time(),
        }
        return result

    def estimate_cost(self) -> float:
        enc = tiktoken.encoding_for_model(self.judge.model)
        num_input_tokens = (
            len(enc.encode(self.question["turns"][0]))
            + len(enc.encode(self.answer_1["choices"][0]["turns"][0]))
            + len(enc.encode(self.answer_2["choices"][0]["turns"][0]))
            + len(enc.encode(self.judge.prompt_template["system_prompt"]))
            + len(enc.encode(self.judge.prompt_template["prompt_template"]))
        )
        if self.ref_answer:
            num_input_tokens += len(
                enc.encode(self.ref_answer["choices"][0]["turns"][0])
            )
        num_output_tokens = 200  # Estimated from a few samples
        if self.judge.model in {"gpt-4", "gpt-4-0613"}:
            return (0.03 * num_input_tokens + 0.06 * num_output_tokens) / 1_000
        elif self.judge.model == "gpt-4-1106-preview":
            return (0.01 * num_input_tokens + 0.03 * num_output_tokens) / 1_000
        elif self.judge.model == "gpt-3.5-turbo":
            return (0.0005 * num_input_tokens + 0.0015 * num_output_tokens) / 1_000
        raise AssertionError

    @staticmethod
    def get_winner(judgment: str, model_a: str, model_b: str) -> str:
        if "[[A]]" in judgment:
            return model_a
        elif "[[B]]" in judgment:
            return model_b
        elif "[[C]]" in judgment:
            return "tie"
        return "error"


def load_questions(question_file: Union[str, Path]) -> list[dict]:
    """Load questions from a file.

    Args:
        question_file (Union[str, Path]): The question file.
    """
    with open(question_file, "r") as fin:
        return [json.loads(line) for line in fin]


def get_model_list(answer_dir: Union[str, Path]):
    """Get model list from answer directory.

    Args:
        answer_dir (Union[str, Path]): The answer directory.
    """
    return [path.name for path in Path(answer_dir).iterdir()]


# def load_model_answers(answer_dir: Union[str, Path]):
#     """Load model answers.

#     Args:
#         answer_dir (Union[str, Path]): The answer directory.
#     """
#     answers = {}
#     with open(Path(answer_dir) / "results.jsonl", "r") as fin:
#         for line in fin:
#             answer = json.loads(line)
#             answers[answer["question_id"]] = answer
#     return answers

def load_model_answers(answer_dir: Union[str, Path]):
    """Load model answers.

    Args:
        answer_dir (Union[str, Path]): The answer directory.
    """
    answers = {}
    with open(Path(answer_dir) / "results.jsonl", "r") as fin:
        for line in fin:
            answer = json.loads(line)
            qid = answer["question_id"]
            if qid not in answers:
                answers[qid] = []
            answers[qid].append(answer)
    return answers


def load_model_config(answer_dir: Union[str, Path]):
    """Load model config.

    Args:
        answer_dir (Union[str, Path]): The answer directory.
    """
    with open(Path(answer_dir) / "config.json", "r") as fin:
        return json.load(fin)


def load_judgements(judgement_dir: Union[str, Path]):
    """Load judgements.

    Args:
        judgement_dir (Union[str, Path]): The judgement directory.
    """
    judgements = {}
    for path in Path(judgement_dir).glob("*.jsonl"):
        with open(path, "r") as fin:
            results = []
            for line in fin:
                results.append(json.loads(line))
            judgements[path.stem] = results
    return judgements


def load_judge_prompts(prompt_file: Union[str, Path]):
    """Load judge prompts.

    Args:
        prompt_file (Union[str, Path]): The prompt file.
    """
    prompts = {}
    with open(prompt_file) as fin:
        for line in fin:
            line = json.loads(line)
            prompts[line["name"]] = line
    return prompts


def filter_single_judgements(
    result_id_results_map: dict[str, list[dict]], model_list: Optional[list[str]] = None
):
    """Filter results by specified models.

    Args:
        result_id_results_map (dict[str, list[dict]]): A dict of results.
        model_list (list[str], optional): A list of models. Defaults to None.
    """
    if model_list is None:
        return result_id_results_map
    filtered_result_id_results_map = {}
    for result_id, results in result_id_results_map.items():
        result = results[0]
        if result["model"] in model_list:
            filtered_result_id_results_map[result_id] = results
    return filtered_result_id_results_map


def filter_pairwise_judgements(
    result_id_results_map: dict[str, list[dict]],
    model_list: Optional[list[str]] = None,
    baseline_model: Optional[str] = None,
):
    """Filter results by specified models.

    Args:
        result_id_results_map (dict[str, list[dict]]): A dict of results.
        model_list (list[str], optional): A list of models. Defaults to None.
        baseline_model (str, optional): The baseline model. Defaults to None.
    """
    filtered_result_id_results_map = {}
    for result_id, results in result_id_results_map.items():
        result = results[0]
        if model_list and baseline_model:
            if (
                result["model_1"] in model_list and result["model_2"] == baseline_model
            ) or (
                result["model_2"] in model_list and result["model_1"] == baseline_model
            ):
                filtered_result_id_results_map[result_id] = results
        elif model_list and baseline_model is None:
            if result["model_1"] in model_list and result["model_2"] in model_list:
                filtered_result_id_results_map[result_id] = results
        elif model_list is None and baseline_model:
            if (
                result["model_1"] == baseline_model
                or result["model_2"] == baseline_model
            ):
                filtered_result_id_results_map[result_id] = results
        else:
            filtered_result_id_results_map[result_id] = results
    return filtered_result_id_results_map