import json
import asyncio
from pathlib import Path

from questionary import select, text
from tqdm.asyncio import tqdm_asyncio


async def run_extraction(args, send_fn, default_concurrency=16, default_output_suffix="_output.jsonl", cleanup_fn=None):
    if args.extract_file:
        extract_file = args.extract_file
    else:
        files = [f.name for f in Path("./PGraphRAG").glob("*.json")]
        extract_file = await select("Select a extract file", choices=files).ask_async()

    all_reviews = load_reviews(f"./PGraphRAG/{extract_file}")
    total_reviews = len(all_reviews)

    default_output = f"./PGraphRAG/output/{extract_file.split('.')[0]}{default_output_suffix}"
    output_path = args.output_path or (await text(
        "Enter output path",
        default=default_output,
    ).ask_async())

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    completed = load_completed_indices(output_path)
    remaining = [(idx, review) for idx, review in enumerate(all_reviews, 1) if idx not in completed]

    if completed:
        print(f"Resuming: {len(completed)} already done, {len(remaining)} remaining")
    print(f"Processing {len(remaining)}/{total_reviews} reviews -> {output_path}")

    concurrency = args.concurrency or default_concurrency
    semaphore = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()

    with tqdm_asyncio(total=len(remaining), desc=f"Review", unit="review") as pbar:
        tasks = [
            process_review(idx, review, send_fn, pbar, output_path, semaphore, write_lock)
            for idx, review in remaining
        ]
        await asyncio.gather(*tasks)

    if cleanup_fn:
        cleanup_fn()

    print(f"\nCompleted! Output saved to {output_path}")


def load_completed_indices(output_path):
    completed = set()
    path = Path(output_path)
    if path.exists():
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    if "idx" in record:
                        completed.add(record["idx"])
                except json.JSONDecodeError:
                    continue
    return completed


def load_reviews(path):
    reviews = []
    with open(path, 'r') as f:
        data = json.load(f)
        for user in data:
            for review in user.get("profile", []):
                reviews.append({
                    "user_id": user["id"],
                    "product_id": review["pid"],
                    "rating": review["rating"],
                    "title": review["title"],
                    "text": review["text"]
                })
    print(f"Loaded {len(reviews)} reviews from {path}")
    return reviews


async def process_review(idx, review, send_fn, pbar, output_path, semaphore, write_lock):
    async with semaphore:
        print(f"\n--- Review {idx} ---")
        print(f"Product: {review['product_id']} | Rating: {review['rating']}")
        print(f"Text: {review['text']}")

        max_retries = 10
        for attempt in range(max_retries):
            try:
                response = await send_fn(f"Text: {review['text']}")
                output = response.choices[0].message.content

                parsed_json = json.loads(output)
                
                if isinstance(parsed_json, list):
                    print(f"Warning: Model returned list instead of dict on attempt {attempt + 1}. Retrying...")
                    continue
                
                if "triples" not in parsed_json:
                    print(f"Warning: Model output missing 'triples' key on attempt {attempt + 1}. Retrying...")
                    continue
                
                parsed_json["idx"] = idx
                parsed_json.update(review)
                async with write_lock:
                    with open(output_path, "a") as f:
                        f.write(json.dumps(parsed_json) + "\n")
                        f.flush()
                
                print(f"Triples:\n{output}")
                break
                
            except json.JSONDecodeError as e:
                print(f"Warning: Model output was not valid JSON on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    print(f"Failed to get valid output for review {idx} after {max_retries} attempts.")
                await asyncio.sleep(0.5)
            except Exception as e:
                print(f"Warning: Error on attempt {attempt + 1}: {type(e).__name__}: {e}")
                if attempt == max_retries - 1:
                    print(f"Failed to process review {idx} after {max_retries} attempts.")
                await asyncio.sleep(0.5)

        pbar.update(1)
