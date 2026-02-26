import torch
from pathlib import Path
from questionary import select
from datasets import load_dataset
from transformers import AutoTokenizer, PreTrainedModel
from transformers.models.ministral3 import Ministral3ForCausalLM
from trl import SFTConfig, SFTTrainer

TRAIN_CONFIGS = {
    "conservative": {
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 32,
        "learning_rate": 5e-6,
        "num_train_epochs": 1,
        "warmup_ratio": 0.15,
        "max_length": 2048,
    },
    "balanced": {
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 16,
        "learning_rate": 2e-5,
        "num_train_epochs": 3,
        "warmup_ratio": 0.1,
        "max_length": 2048,
    },
    "aggressive": {
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 8,
        "learning_rate": 5e-5,
        "num_train_epochs": 5,
        "warmup_ratio": 0.05,
        "max_length": 2048,
    },
}

async def finetune(args):
    model_id = args.model_path or "models/Ministral-3-3B-Instruct-2512"

    train_dir = Path("./train")
    if args.dataset:
        dataset_path = args.dataset
    else:
        files = [f.name for f in train_dir.glob("*.jsonl")]
        chosen = await select("Select training dataset", choices=files).ask_async()
        dataset_path = str(train_dir / chosen)
    output_dir = args.output_path or "ministral-3b-pgraphrag-fft"
    final_dir = f"{output_dir}-final"

    config_name = args.train_config or (await select(
        "Select training config",
        choices=list(TRAIN_CONFIGS.keys()),
    ).ask_async())
    config = TRAIN_CONFIGS[config_name]
    print(f"Using '{config_name}' training config")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model: PreTrainedModel = Ministral3ForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    dataset = load_dataset("json", data_files=dataset_path, split="train")

    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        lr_scheduler_type="cosine",
        warmup_ratio=config["warmup_ratio"],
        num_train_epochs=config["num_train_epochs"],
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        weight_decay=0.01,
        seed=3407,
        dataset_text_field="text",
        max_length=config["max_length"],
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    trainer.train()

    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)