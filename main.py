from questionary import select
from engine import Engine
import asyncio
from engine import CloudEngine
from utils import run_extraction
from finetune import finetune
import argparse

def parse_args(task_choices, device_choices):
    parser = argparse.ArgumentParser(description="Knowledge Graph CLI using MLC-LLM")
    parser.add_argument("--task", choices=task_choices)
    parser.add_argument("--device", choices=device_choices)
    parser.add_argument("--extract-file")
    parser.add_argument("--output-path")
    parser.add_argument("--model-path")
    parser.add_argument("--dataset")
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--train-config", choices=["conservative", "balanced", "aggressive"])
    return parser.parse_args()

async def main():
    # choices
    task_choices = ["extract-triples", "extract-finetune", "finetune", "normalize-data", "query"]
    device_choices = ["cuda", "metal", "cpu"]

    args = parse_args(task_choices=task_choices, device_choices=device_choices)

    task = args.task or (await select(
        "Knowledge Graph CLI",
        choices=task_choices,
    ).ask_async())

    if task == task_choices[0]:
        device = args.device or (await select(
            "Select a device",
            choices=["cuda", "metal", "cpu"],
        ).ask_async())

        model_path = args.model_path or "./models/Ministral-3-3B-Instruct-2512-BF16-q4f16_1-MLC"
        engine = Engine(model_path=model_path, device=device)

        await run_extraction(
            args,
            send_fn=engine.send_extract_message,
            default_concurrency=4,
            cleanup_fn=engine.terminate,
        )
    elif task == task_choices[1]:
        engine = CloudEngine()

        async def send_fn(message):
            return await asyncio.to_thread(engine.send_extract_message, message)

        await run_extraction(
            args,
            send_fn=send_fn,
            default_concurrency=16,
            default_output_suffix="_finetune_output.jsonl",
        )
    elif task == task_choices[2]:
        await finetune(args)
    elif task == task_choices[3]:
        print("TODO")
    elif task == task_choices[4]:
        print("TODO")


if __name__ == "__main__":
    asyncio.run(main())