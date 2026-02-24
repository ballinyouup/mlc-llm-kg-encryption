import argparse
import asyncio
import json
from pathlib import Path

from tqdm.asyncio import tqdm_asyncio
from data_process import load_reviews
from engine import Engine


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-splits", type=int, default=4, help="Total number of dataset splits")
    parser.add_argument("--split-id", type=int, required=True, help="Which split this instance processes (0..num_splits-1)")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Engine device backend (e.g. metal, cuda, vulkan, cpu)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./PGraphRAG/output"),
        help="Directory for output jsonl files",
    )
    return parser.parse_args()


def get_split_reviews(reviews, num_splits, split_id):
    # Round-robin shard for better balancing
    # global_idx is 1-based index in full dataset
    return [
        (global_idx, review)
        for global_idx, review in enumerate(reviews, 1)
        if (global_idx - 1) % num_splits == split_id
    ]


async def process_review(local_idx, global_idx, review, total_split_reviews, engine, pbar, output_path):
    print(f"\n--- Split Review {local_idx}/{total_split_reviews} (global #{global_idx}) ---")
    print(f"Product: {review['product_id']} | Rating: {review['rating']}")
    print(f"Text: {review['text']}")

    response = await engine.send_extract_message(f"Text: {review['text']}")
    output = response.choices[0].message.content

    try:
        parsed_json = json.loads(output)
        parsed_json["user_id"] = review["user_id"]
        parsed_json["product_id"] = review["product_id"]
        with open(output_path, "a") as f:
            f.write(json.dumps(parsed_json) + "\n")
    except json.JSONDecodeError:
        print(f"Warning: Model output for global review {global_idx} was not valid JSON.")

    print(f"Triples:\n{output}")
    pbar.update(1)


async def main():
    args = parse_args()

    if args.num_splits <= 0:
        raise ValueError("--num-splits must be > 0")
    if not (0 <= args.split_id < args.num_splits):
        raise ValueError("--split-id must be in range [0, num_splits - 1]")

    engine = Engine(device=args.device, model_choice=Engine.model_choice.mistral_7b)

    all_reviews = load_reviews(Path("./PGraphRAG/amazon_train.json"))
    split_reviews = get_split_reviews(all_reviews, args.num_splits, args.split_id)
    total_split_reviews = len(split_reviews)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"amazon_train_output_split_{args.split_id}.jsonl"

    print(
        f"Processing split {args.split_id}/{args.num_splits - 1}: "
        f"{total_split_reviews} reviews -> {output_path}"
    )

    with tqdm_asyncio(total=total_split_reviews, desc=f"Split {args.split_id}", unit="review") as pbar:
        tasks = [
            process_review(local_idx, global_idx, review, total_split_reviews, engine, pbar, output_path)
            for local_idx, (global_idx, review) in enumerate(split_reviews, 1)
        ]
        await asyncio.gather(*tasks)

    engine.terminate()


asyncio.run(main())