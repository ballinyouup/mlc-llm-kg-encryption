from questionary import select, text
import asyncio
from pathlib import Path

from tqdm.asyncio import tqdm_asyncio
from engine import Engine, ModelChoice
from extract import process_review, load_reviews

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
    task_choices = ["extract-triples", "query"]
    model_choices = [model.name for model in ModelChoice]
    device_choices = ["cuda", "metal", "cpu"]

    args = parse_args(task_choices=task_choices, model_choices=model_choices, device_choices=device_choices)

    task = args.task or (await select(
        "Knowledge Graph CLI",
        choices=task_choices,
    ).ask_async())

    if task == task_choices[0]:
        model_name = args.model or (await select(
            "Select a model",
            choices=model_choices,
        ).ask_async())
        model_choice = ModelChoice[model_name]

        # Device
        device = args.device or (await select(
            "Select a device",
            choices=["cuda", "metal", "cpu"],
        ).ask_async())

        engine = Engine(device=device, model_choice=model_choice)

        # Extract file
        if args.extract_file:
            extract_file = args.extract_file
        else:
            files = [f.name for f in Path("./PGraphRAG").glob("*.json")]
            extract_file = await select("Select a extract file", choices=files).ask_async()


        all_reviews = load_reviews(f"./PGraphRAG/{extract_file}")
        total_reviews = len(all_reviews)

        # Output path
        output_path = args.output_path or (await text(
            "Enter output path",
            default=f"./PGraphRAG/output/{extract_file}_output.jsonl",
        ).ask_async())

        print(f"Processing {total_reviews} reviews -> {output_path}")

        # progress bar and async processing of reviews
        with tqdm_asyncio(total=total_reviews, desc=f"Review", unit="review") as pbar:
            tasks = [
                process_review(idx, review, engine, pbar, output_path)
                for idx, review in enumerate(all_reviews, 1)
            ]
            await asyncio.gather(*tasks)

        engine.terminate()
    elif task == task_choices[1]:
        print("TODO")


if __name__ == "__main__":
    asyncio.run(main())