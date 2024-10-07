import argparse
import json
import logging
import os
import time

import openai
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
import shortuuid
from common import PREDICTION_DIR, QUESTION_FILE, load_questions
from dotenv import load_dotenv
from tqdm import tqdm

logger = logging.getLogger(__name__)

load_dotenv()
# TODO: The 'openai.organization' option isn't read in the client API. You will need to pass it when you instantiate the client, e.g. 'OpenAI(organization=os.getenv("OPENAI_ORGANIZATION"))'
# openai.organization = os.getenv("OPENAI_ORGANIZATION")


def generate_response(input_text, generation_config) -> str:
    """Generate a response from the input text.

    Args:
        input_text: The input text.
        generation_config: The config for the generation.
    """
    response = client.completions.create(prompt=input_text, **generation_config)
    return response.choices[0].text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to configuration file"
    )
    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="Verbosity level"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite the existing results"
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

    logger.info(f"Loading config from {args.config}")
    with open(args.config, "r") as f:
        config = json.load(f)
    logger.debug(config)

    logger.info("Load the data")
    questions = load_questions(str(QUESTION_FILE))

    logger.info("Start inference.")
    model_id = config["model_id"]
    if "generation_config" not in config:
        raise ValueError("'generation_config' is not found in the config file.")
    prompt_template = config["prompt_template"]
    if "{instruction}" not in prompt_template:
        raise ValueError("prompt_template must contain {instruction}")
    generation_config = config["generation_config"]

    prediction_dir = PREDICTION_DIR / model_id
    prediction_file = prediction_dir / "results.jsonl"
    if prediction_file.exists() and not args.overwrite:
        raise FileExistsError(
            f"{prediction_file} already exists. Use --overwrite to overwrite."
        )

    results = []
    for question in tqdm(questions):
        instruction = question["turns"][0]
        output = generate_response(
            input_text=prompt_template.format_map({"instruction": instruction}),
            generation_config=generation_config,
        )

        logger.debug(f"{instruction}\n\n{output}")

        results.append(
            {
                "question_id": question["question_id"],
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": [{"index": 0, "turns": output}],
                "tstamp": time.time(),
            }
        )

    logger.info("Save the results")
    prediction_dir.mkdir(parents=True, exist_ok=True)
    with open(prediction_file, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    logger.info(f"Saved the results to {prediction_file}")

    logger.info("Save the config")
    config_file = prediction_dir / "config.json"
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved the config to {config_file}")
