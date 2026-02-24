from questionary import select
import asyncio
from engine import ModelChoice
from extract import extract_triples

import argparse

def parse_args(task_choices, model_choices, device_choices):
    parser = argparse.ArgumentParser(description="Knowledge Graph CLI using MLC-LLM")
    parser.add_argument("--task", choices=task_choices)
    parser.add_argument("--model", choices=model_choices)
    parser.add_argument("--device", choices=device_choices)
    parser.add_argument("--extract-file")
    parser.add_argument("--output-path")
    return parser.parse_args()

async def main():
    # choices
    task_choices = ["extract-triples", "normalize-data", "query"]
    model_choices = [model.name for model in ModelChoice]
    device_choices = ["cuda", "metal", "cpu"]

    args = parse_args(task_choices=task_choices, model_choices=model_choices, device_choices=device_choices)

    task = args.task or (await select(
        "Knowledge Graph CLI",
        choices=task_choices,
    ).ask_async())

    if task == task_choices[0]:
        await extract_triples(args, model_choices)
    elif task == task_choices[1]:
        print("TODO")
    elif task == task_choices[2]:
        print("TODO")


if __name__ == "__main__":
    asyncio.run(main())