import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from typing import Optional, Dict, List

from common import (
    JUDGEMENT_DIR,
    JUDGEMENT_PROMPT_FILE,
    NEED_REF_CATS,
    PREDICTION_DIR,
    QUESTION_FILE,
    REFERENCE_DIR,
    Judge,
    MatchPair,
    MatchSingle,
    get_model_list,
    load_judge_prompts,
    load_model_answers,
    load_questions,
)
from tqdm import tqdm
from upload_result import upload_results

logger = logging.getLogger(__name__)


def make_match_groups_single(
    questions: List[dict],
    model_answers: Dict[str, Dict[int, List[dict]]],
    ref_answers: Dict[str, Dict[int, List[dict]]],
    judge_default: Judge,
    judge_math: Judge,
    num_answers_per_question: Optional[int] = None,
):
    """Make match groups for single answer grading."""

    match_groups = {model: [] for model in model_answers}

    for question in questions:
        qid = question["question_id"]
        category = question["category"]

        # Determine if reference answer is needed
        if category in NEED_REF_CATS:
            judge = judge_math
            ref_answer_list = ref_answers[judge.model].get(qid)
            if not ref_answer_list:
                logger.warning(f"No reference answer for question {qid} in model {judge.model}")
                ref_answer = None
            else:
                ref_answer = ref_answer_list[0]
        else:
            judge = judge_default
            ref_answer = None
        # Get all models that have answers for this question
        available_models = [model for model, answers in model_answers.items() if qid in answers]

        for model in available_models:
            answers = model_answers[model][qid]
            if num_answers_per_question is not None:
                selected_answers = answers[:num_answers_per_question]
            else:
                selected_answers = answers

            for answer in selected_answers:
                match = MatchSingle(
                    question=question,
                    model=model,
                    answer=answer,
                    judge=judge,
                    ref_answer=ref_answer,
                )
                match_groups[model].append(match)

    return match_groups


def make_match_groups_pairwise(
    questions: List[dict],
    model_answers: Dict[str, Dict[int, List[dict]]],
    ref_answers: Dict[str, Dict[int, List[dict]]],
    judge_default: Judge,
    judge_math: Judge,
    baseline_model: Optional[str] = None,
    num_answers_per_question: Optional[int] = None,
):
    """Make match groups for pairwise comparison."""

    match_groups = {}

    for question in questions:
        qid = question["question_id"]
        category = question["category"]

        # Determine if reference answer is needed
        if category in NEED_REF_CATS:
            judge = judge_math
            ref_answer_list = ref_answers[judge.model].get(qid)
            if not ref_answer_list:
                logger.warning(f"No reference answer for question {qid} in model {judge.model}")
                ref_answer = None
            else:
                ref_answer = ref_answer_list[0]
        else:
            judge = judge_default
            ref_answer = None

        # Get all models that have answers for this question
        available_models = [model for model, answers in model_answers.items() if qid in answers]

        if baseline_model:
            if baseline_model not in available_models:
                logger.warning(f"Baseline model {baseline_model} does not have an answer for question {qid}. Skipping.")
                continue
            non_baseline_models = [model for model in available_models if model != baseline_model]
        else:
            non_baseline_models = available_models

        if num_answers_per_question is not None:
            selected_non_baseline_models = non_baseline_models[:num_answers_per_question]
        else:
            selected_non_baseline_models = non_baseline_models

        if baseline_model:
            selected_models = selected_non_baseline_models + [baseline_model]
        else:
            selected_models = selected_non_baseline_models

        # Generate all unique pairs
        for model_1, model_2 in combinations(selected_models, 2):
            if baseline_model and (model_1 != baseline_model and model_2 != baseline_model):
                # In pairwise-baseline mode, only create pairs with the baseline
                continue

            pair_key = f"pairwise:{model_1}_{model_2}"
            if pair_key not in match_groups:
                match_groups[pair_key] = []

            answers_1 = model_answers[model_1][qid]
            answers_2 = model_answers[model_2][qid]

            if num_answers_per_question is not None:
                selected_answers_1 = answers_1[:num_answers_per_question]
                selected_answers_2 = answers_2[:num_answers_per_question]
            else:
                selected_answers_1 = answers_1
                selected_answers_2 = answers_2

            for ans1 in selected_answers_1:
                for ans2 in selected_answers_2:
                    match = MatchPair(
                        question=question,
                        model_1=model_1,
                        model_2=model_2,
                        answer_1=ans1,
                        answer_2=ans2,
                        judge=judge,
                        ref_answer=ref_answer,
                    )
                    match_groups[pair_key].append(match)

    return match_groups

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="pairwise-baseline",
        choices=["pairwise-baseline", "pairwise-all", "single"],
        help=(
            "Evaluation mode. "
            "`pairwise-baseline` runs pairwise comparison against a baseline. "
            "`pairwise-all` runs pairwise comparison between all pairs. "
            "`single` runs single answer grading."
        ),
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="gpt-4",
        choices=["gpt-4", "gpt-4-0613", "gpt-4-1106-preview", "gpt-3.5-turbo"],
        help="The judge model.",
    )
    parser.add_argument(
        "--baseline-model",
        type=str,
        default="openai--text-davinci-003",
        help="The baseline model. This is only used in `pairwise-baseline` mode.",
    )
    parser.add_argument(
        "--model-list",
        type=str,
        nargs="+",
        default=None,
        help="A list of models to be evaluated. If not specified, all models will be evaluated",
    )
    parser.add_argument(
        "--parallel", type=int, default=1, help="The number of concurrent API calls."
    )
    parser.add_argument("--first-n", type=int, help="Only run the first `n` judgments.")
    parser.add_argument(
        "--yes", "-y", action="store_true", help="Skip confirmation and run."
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing judgment files."
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log to wandb.",
    )
    parser.add_argument(
        "--verbose", "-v", action="count", default=0, help="Verbosity level"
    )
    parser.add_argument(
        "--num_answers_per_question",
        type=int,
        default=None,
        help="Number of answers to evaluate per question.",
    )
    args = parser.parse_args()

    if args.verbose == 0:
        level = logging.INFO
    else:
        level = logging.DEBUG
    logging.basicConfig(
        format="| %(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=level,
    )

    if args.wandb:
        import wandb

        wandb.login()
        if args.mode != "pairwise-baseline":
            logger.warning(
                "Leaderboard is only available in pairwise-baseline mode. "
                "Only raw outputs will be logged."
            )

    logger.info("Load questions")
    questions = load_questions(QUESTION_FILE)
    if args.first_n:
        logger.warning(f"Only run the first {args.first_n} judgments")
        questions = questions[: args.first_n]

    logger.info("Load answers")
    if args.model_list is None:
        models = get_model_list(PREDICTION_DIR)
    else:
        models = args.model_list
        if args.mode == "pairwise-baseline" and args.baseline_model not in models:
            models.append(args.baseline_model)
    model_answers = {}
    for model in sorted(models):
        answers = load_model_answers(PREDICTION_DIR / model)
        for question in questions:
            qid = question["question_id"]
            if qid not in answers:
                logger.error(f"Question ID {qid} missing in model {model} answers.")
                raise ValueError(f"Question ID {qid} missing in model {model} answers.")
        model_answers[model] = answers

    logger.info("Load reference answers")
    judge_model = args.judge_model
    answers = load_model_answers(REFERENCE_DIR / "gpt-4")
    for question in filter(lambda x: x["category"] in NEED_REF_CATS, questions):
        qid = question["question_id"]
        if qid not in answers:
            logger.error(f"Reference answer for question ID {qid} missing.")
            raise ValueError(f"Reference answer for question ID {qid} missing.")
    ref_answers = {judge_model: answers}

    logger.info("Load judge prompts")
    judge_prompts = load_judge_prompts(JUDGEMENT_PROMPT_FILE)

    logger.info("Make matches")
    if args.mode == "single":
        match_groups = make_match_groups_single(
            questions,
            model_answers,
            ref_answers=ref_answers,
            judge_default=Judge(args.judge_model, judge_prompts["single"]),
            judge_math=Judge(args.judge_model, judge_prompts["single-math"]),
            num_answers_per_question=args.num_answers_per_question,
        )
        output_dir = JUDGEMENT_DIR / "single" / args.judge_model
    else:
        assert args.mode in {"pairwise-baseline", "pairwise-all"}
        if args.mode == "pairwise-all":
            baseline_model = None
        else:
            baseline_model = args.baseline_model
        match_groups = make_match_groups_pairwise(
            questions,
            model_answers,
            ref_answers=ref_answers,
            judge_default=Judge(args.judge_model, judge_prompts["pair"]),
            judge_math=Judge(args.judge_model, judge_prompts["pair-math"]),
            baseline_model=baseline_model,
            num_answers_per_question=args.num_answers_per_question,
        )
        output_dir = JUDGEMENT_DIR / "pairwise" / args.judge_model

    # Filter out existing match_ids if not overwriting
    target_match_ids = set()
    for match_id in match_groups:
        output_file = output_dir / f"{match_id}.jsonl"
        if output_file.exists():
            if not args.overwrite:
                logger.info(f"Skip {match_id}; to overwrite, use --overwrite")
                continue
        target_match_ids.add(match_id)
    match_groups = {k: v for k, v in match_groups.items() if k in target_match_ids}

    logger.info(f"Mode: {args.mode}")
    logger.info(f"Judge model: {args.judge_model}")
    if args.mode == "pairwise-baseline":
        logger.info(f"Baseline model: {args.baseline_model}")
    logger.info(f"Total number of questions: {len(questions):,}")
    logger.info(
        f"Total number of matches: {sum(len(matches) for matches in match_groups.values()):,}"
    )
    estimated_cost = 0
    for matches in match_groups.values():
        estimated_cost += sum(m.estimate_cost() for m in matches)
    logger.info(f"Total cost (estimated): ${int(estimated_cost):,}")
    logger.info(f"Output directory: {output_dir}")

    if not args.yes:
        input("Press Enter to confirm...")

    logger.info("Play matches")
    for match_id, matches in match_groups.items():
        output_file = output_dir / f"{match_id}.jsonl"
        results = []
        with ThreadPoolExecutor(args.parallel) as executor:
            futures = [executor.submit(match.play) for match in matches]
            for future in tqdm(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing match {match_id}: {e}")

        logger.info(f"Write {len(results)} judgments")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        logger.info(f"Saved the judgments to {output_file}")

        if args.wandb:
            logger.info("Log to wandb")
            upload_results(args.mode, match_id, results, args.baseline_model)