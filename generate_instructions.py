import logging
import time
import json
import os
import random
import re
import string
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Optional, List
from multiprocessing.pool import ThreadPool

import numpy as np
import openai
import tiktoken
import tqdm
from openai.error import AuthenticationError, InvalidRequestError
from rouge_score import rouge_scorer

import fire
from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_not_exception_type


def write_json(json_object: object, file_path: str):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(json_object, f, indent=4, default=str)


def read_json(path: str) -> object:
    with open(path, "r", encoding="utf-8") as user_file:
        parsed_json = json.load(user_file)
    return parsed_json


def log_cost(responses):
    prompt_tokens = sum(
        [response["usage"]["prompt_tokens"] for response in responses]
    )
    completion_tokens = sum(
        [response["usage"]["completion_tokens"] for response in responses]
    )
    cost = (prompt_tokens * 0.0015 + completion_tokens * 0.002) / 1000
    logging.info(f"prompt_tokens={prompt_tokens}, completion_tokens={completion_tokens}, cost={cost}")


def count_tokens(prompt: str, model: str = "gpt-3.5-turbo") -> int:
    return len(tiktoken.encoding_for_model(model).encode(prompt))


@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(20),
    retry=retry_if_not_exception_type((InvalidRequestError, AuthenticationError, TypeError))
)
def openai_chat_request(
        user_prompt: str,
        model: str = "gpt-3.5-turbo",
        max_tokens: Optional[int] = None,
        temperature: float = 0,
        top_p: float = 1,
        stop: Optional[List[str]] = None,
):
    logging.debug(f"Making a request with {count_tokens(user_prompt)} tokens")

    completion = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": user_prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        n=1,
        stop=stop
    )
    completion["input_prompt"] = user_prompt
    return completion


INSTRUCTION_TXT = "Juhis"
INPUT_TXT = "Sisend"
OUTPUT_TXT = "Väljund"


def encode_prompt(prompt_prefix: str, prompt_instructions: List[dict]):
    """Encode multiple prompt instructions into a single string."""
    prompt = prompt_prefix

    for idx, task_dict in enumerate(prompt_instructions):
        (instruction, input, output) = task_dict["instruction"], task_dict["input"], task_dict["output"]
        instruction = re.sub(r"\s+", " ", instruction).strip().rstrip(":")
        input = "<noinput>" if input.lower() == "" else input
        prompt += f"###\n"
        prompt += f"{idx + 1}. {INSTRUCTION_TXT}: {instruction}\n"
        prompt += f"{idx + 1}. {INPUT_TXT}:\n{input}\n"
        prompt += f"{idx + 1}. {OUTPUT_TXT}:\n{output}\n"
    prompt += f"###\n"
    prompt += f"{idx + 2}. {INSTRUCTION_TXT}:\n"
    return prompt


def post_process_gpt_response(num_prompt_instructions, response):
    if response is None:
        return []
    raw_instructions = f"{num_prompt_instructions + 1}. {INSTRUCTION_TXT}:"
    raw_instructions += response["choices"][0]["message"]["content"]
    raw_instructions = re.split("###", raw_instructions)
    instructions = []
    for idx, inst in enumerate(raw_instructions):
        # if the decoding stops due to length, the last example is likely truncated so we discard it
        if idx == len(raw_instructions) - 1 and response["choices"][0]["finish_reason"] == "length":
            logging.info("Decoding stopped due to length, discarding last example.")
            continue
        idx += num_prompt_instructions + 1
        splitted_data = re.split(f"{idx}\.\s+({INSTRUCTION_TXT}|{INPUT_TXT}|{OUTPUT_TXT}):", inst)
        if len(splitted_data) != 7:
            logging.info(f"Error in splitting data (Instruction {idx}): {splitted_data}")
            continue
        else:
            inst = splitted_data[2].strip()
            input = splitted_data[4].strip()
            input = "" if input.lower() == "<noinput>" else input
            output = splitted_data[6].strip()
        # filter out too short or too long instructions
        if len(inst.split()) <= 3 or len(inst.split()) > 150:
            logging.info(f"Instruction {idx} too short or too long.")
            continue
        # filter based on keywords that are not suitable for language models.
        blacklist = [
            "pilt",
            "pildid",
            "joonista",
            "joonistus",
            "joonistused",
            "joonesta",
            "video",
            "audio",
            "diagram",
        ]
        blacklist += []
        if any(find_word_in_string(word, inst) for word in blacklist):
            logging.info(f"Instruction {idx} contains blacklisted words: {inst}")
            continue

        # filter those starting with punctuation
        if inst[0] in string.punctuation:
            logging.info(f"Instruction {idx} starts with punctuation: {inst}")
            continue

        instructions.append({"instruction": inst, "input": input, "output": output})
    return instructions


def find_word_in_string(w, s):
    return re.compile(r"\b({0})\b".format(w), flags=re.IGNORECASE).search(s)


def generate_instruction_following_data(
        output_dir="./",
        seed_tasks_path="data/seed_tasks_est.jsonl",
        prompt_path="data/prompt_est.txt",
        num_instructions_to_generate=10,
        model_name="gpt-3.5-turbo",
        num_prompt_instructions=3,
        temperature=1.0,
        top_p=1.0,
        num_cpus=16,
        num_parallel_requests=1,

        out_file_name="instructions.json",
        response_file_name_prefix="responses",
        out_jsonl_file_name_prefix="instructions",
):
    if num_parallel_requests < 1:
        raise ValueError("num_parallel_requests must be >= 1")
    if num_cpus < 1:
        raise ValueError("num_cpus must be >= 1")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    seed_tasks = [json.loads(l) for l in open(seed_tasks_path, "r")]
    seed_instruction_data = [
        {"instruction": t["instruction"], "input": t["instances"][0]["input"], "output": t["instances"][0]["output"]}
        for t in seed_tasks
    ]
    logging.info(f"Loaded {len(seed_instruction_data)} human-written seed instructions")
    responses = []

    os.makedirs(output_dir, exist_ok=True)
    request_idx = 0

    # load the LM-generated instructions
    machine_instruction_data = []
    if os.path.exists(os.path.join(output_dir, out_file_name)):
        machine_instruction_data = read_json(os.path.join(output_dir, out_file_name))
        assert isinstance(machine_instruction_data, list)
        logging.info(f"Loaded {len(machine_instruction_data)} machine-generated instructions")

    start_len = len(machine_instruction_data)

    # similarities = {}
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    # now let's generate new instructions!
    progress_bar = tqdm.tqdm(total=num_instructions_to_generate)
    if machine_instruction_data:
        progress_bar.update(len(machine_instruction_data))

    # first we tokenize all the seed instructions and generated machine instructions
    all_instructions = [d["instruction"] for d in seed_instruction_data] + [
        d["instruction"] for d in machine_instruction_data
    ]
    all_instruction_tokens = [scorer._tokenizer.tokenize(inst) for inst in all_instructions]
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_prefix = f.read() + "\n"

    logging.info("Max tokens in prompt: " +
                 str(max(
                     count_tokens(encode_prompt(prompt_prefix, [data] * num_prompt_instructions))
                     for data in seed_instruction_data
                 ))
                 )
    jsonl_out_file = open(os.path.join(output_dir, f"{out_jsonl_file_name_prefix}_{start_len}.jsonl"), "w")
    try:
        while len(machine_instruction_data) < num_instructions_to_generate:
            request_idx += 1
            prompts = [
                encode_prompt(prompt_prefix, random.sample(seed_instruction_data, num_prompt_instructions))
                for _ in range(num_parallel_requests)
            ]
            request_start = time.time()
            with ThreadPool(num_parallel_requests) as p:
                results = p.map(
                    lambda x: openai_chat_request(
                        x,
                        model=model_name,
                        max_tokens=None,
                        top_p=top_p,
                        temperature=temperature,
                    ),
                    prompts,
                )
            responses.extend(results)
            request_duration = time.time() - request_start
            logging.debug(f"Request {request_idx}:\n{results}")

            process_start = time.time()
            instruction_data = []
            for result in results:
                instruction_data += post_process_gpt_response(num_prompt_instructions, result)

            total = len(instruction_data)
            keep = 0
            for instruction_data_entry in instruction_data:
                # computing similarity with the pre-tokenzied instructions
                new_instruction_tokens = scorer._tokenizer.tokenize(instruction_data_entry["instruction"])
                with Pool(num_cpus) as p:
                    rouge_scores = p.map(
                        partial(rouge_scorer._score_lcs, new_instruction_tokens),
                        all_instruction_tokens,
                    )
                rouge_scores = [score.fmeasure for score in rouge_scores]
                most_similar_instructions = {
                    all_instructions[i]: rouge_scores[i] for i in np.argsort(rouge_scores)[-10:][::-1]
                }
                if max(rouge_scores) > 0.7:
                    logging.info(
                        f"Instruction {instruction_data_entry['instruction']} too similar to existing instructions")
                    continue
                else:
                    keep += 1
                instruction_data_entry["most_similar_instructions"] = most_similar_instructions
                instruction_data_entry["avg_similarity_score"] = float(np.mean(rouge_scores))

                machine_instruction_data.append(instruction_data_entry)
                jsonl_out_file.write(json.dumps(instruction_data_entry) + "\n")
                jsonl_out_file.flush()

                logging.info(f"Adding instruction {instruction_data_entry}")
                all_instructions.append(instruction_data_entry["instruction"])
                all_instruction_tokens.append(new_instruction_tokens)
                progress_bar.update(1)
            process_duration = time.time() - process_start
            logging.info(f"Request {request_idx} took {request_duration:.2f}s, processing took {process_duration:.2f}s")
            logging.info(f"Generated {total} instructions, kept {keep} instructions")
        log_cost(responses)
    finally:
        jsonl_out_file.close()
        write_json(machine_instruction_data, os.path.join(output_dir, out_file_name))
        write_json(responses, os.path.join(output_dir,
                                           f"{response_file_name_prefix}_{start_len}_{len(machine_instruction_data)}.json"))


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    openai.api_key = os.environ["OPENAI_API_KEY"]
    fire.Fire(generate_instruction_following_data)
