import argparse
import asyncio
import json
from pathlib import Path

from tqdm.asyncio import tqdm_asyncio
from data_process import load_reviews
from engine import Engine, ModelChoice


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
        "--model",
        type=str,
        default="mistral_7b",
        choices=["ministral_3b", "phi_4_mini", "qwen3_4b", "mistral_7b", "hermes_3_llama_3_2_3b", "llama_3_2_3b"],
        help="Model to use for extraction",
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

    max_retries = 10
    for attempt in range(max_retries):
        response = await engine.send_extract_message(f"Text: {review['text']}")
        output = response.choices[0].message.content

        try:
            parsed_json = json.loads(output)
            
            # Handle case where model returns a list instead of dict
            if isinstance(parsed_json, list):
                print(f"Warning: Model returned list instead of dict on attempt {attempt + 1}. Retrying...")
                continue
            
            # Handle case where model returns dict without "triples" key
            if "triples" not in parsed_json:
                print(f"Warning: Model output missing 'triples' key on attempt {attempt + 1}. Retrying...")
                continue
            
            # Success - add metadata and write
            parsed_json["user_id"] = review["user_id"]
            parsed_json["product_id"] = review["product_id"]
            with open(output_path, "a") as f:
                f.write(json.dumps(parsed_json) + "\n")
            
            print(f"Triples:\n{output}")
            break
            
        except json.JSONDecodeError:
            print(f"Warning: Model output was not valid JSON on attempt {attempt + 1}.")
            if attempt == max_retries - 1:
                print(f"Failed to get valid output for global review {global_idx} after {max_retries} attempts.")
        except (TypeError, KeyError) as e:
            print(f"Warning: Unexpected error on attempt {attempt + 1}: {e}")
            if attempt == max_retries - 1:
                print(f"Failed to process global review {global_idx} after {max_retries} attempts.")

    pbar.update(1)


async def main():
    args = parse_args()

    if args.num_splits <= 0:
        raise ValueError("--num-splits must be > 0")
    if not (0 <= args.split_id < args.num_splits):
        raise ValueError("--split-id must be in range [0, num_splits - 1]")

    model_choice = ModelChoice[args.model]
    engine = Engine(device=args.device, model_choice=model_choice)

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