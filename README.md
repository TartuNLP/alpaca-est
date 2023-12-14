# Alpaca-est

Alpaca-est is an instruction dataset generated for Estonian with *gpt-3.5-turbo-0613*, following Alpaca 
([https://github.com/tatsu-lab/stanford_alpaca/tree/main](https://github.com/tatsu-lab/stanford_alpaca/tree/main)).

[`alpaca_est.json`](./data/alpaca_est.json) contains 52,006 instruction-following examples.
Seed tasks and prompt used to generate the dataset are in [`seed_tasks_est.jsonl`](./data/seed_tasks_est.jsonl) and 
[`prompt_est.txt`](./data/prompt_est.txt) respectively.

Example of generating the dataset using [`generate_instructions.py`](generate_instructions.py):
```
python -u generate_instruction.py \
  --output-dir data \
  --seed-tasks-path data/seed_tasks_est.jsonl \
  --prompt-path data/alpaca-seed/prompt_est.txt \
  --num-instructions-to-generate 52000 \
  --num-prompt-instructions 3 --num-cpus 16 --num-parallel-requests 8
```