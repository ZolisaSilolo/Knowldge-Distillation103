"""
Stage B — Student SFT (Supervised Fine-Tuning)

Fine-tunes Qwen2.5-0.5B on clinical data (same data as teacher + teacher-generated
responses) using Unsloth + LoRA. This gives the student a foundation before
the harder KL distillation in Stage A.

Usage:
    python stage_b/train.py
    python stage_b/train.py --dry-run
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import load_config
from utils.checkpoint import upload_checkpoint, upload_metrics
from utils.notify import notify_stage_start, notify_stage_complete, notify_stage_error


STAGE_NAME = "stage_b"


def train(dry_run: bool = False):
    """
    Run Stage B: Student SFT training.
    
    Args:
        dry_run: If True, validate config and data loading only.
    """
    config = load_config(f"{STAGE_NAME}/config.yaml")
    model_cfg = config["model"]
    lora_cfg = config["lora"]
    train_cfg = config["training"]
    data_cfg = config["data"]
    s3_cfg = config["s3"]

    print("=" * 60)
    print(f"ClinIQ — {STAGE_NAME.upper()}: Student SFT")
    print(f"Model: {model_cfg['name']}")
    print(f"LoRA: r={lora_cfg['r']}, alpha={lora_cfg['alpha']}")
    print(f"Epochs: {train_cfg['epochs']}, LR: {train_cfg['learning_rate']}")
    print("=" * 60)

    if dry_run:
        print("\n🔍 DRY RUN — validating config and data...")
        _validate_data(data_cfg)
        print("✅ Dry run complete.")
        return

    from unsloth import FastLanguageModel
    from transformers import TrainingArguments
    from trl import SFTTrainer
    from data.dataset import load_sft_dataset

    notify_stage_start(STAGE_NAME, train_cfg)

    try:
        # 1. Load student model with Unsloth
        print("\n📦 Loading student model with Unsloth...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_cfg["name"],
            max_seq_length=model_cfg["max_seq_length"],
            dtype=None,
            load_in_4bit=model_cfg["load_in_4bit"],
        )

        # 2. Apply LoRA (smaller rank for student)
        print("🔧 Applying LoRA adapters (r=8 for student)...")
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["alpha"],
            lora_dropout=lora_cfg["dropout"],
            target_modules=lora_cfg["target_modules"],
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )

        # 3. Load datasets
        print("📂 Loading training data...")
        train_dataset = load_sft_dataset(
            data_path=str(PROJECT_ROOT / data_cfg["train_path"]),
            tokenizer=tokenizer,
            max_length=data_cfg["max_length"],
            split="train",
        )
        eval_dataset = load_sft_dataset(
            data_path=str(PROJECT_ROOT / data_cfg["eval_path"]),
            tokenizer=tokenizer,
            max_length=data_cfg["max_length"],
            split="eval",
        )

        # 4. Training arguments
        output_dir = str(PROJECT_ROOT / train_cfg["output_dir"])
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=train_cfg["epochs"],
            per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
            gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
            learning_rate=train_cfg["learning_rate"],
            weight_decay=train_cfg["weight_decay"],
            warmup_ratio=train_cfg["warmup_ratio"],
            lr_scheduler_type=train_cfg["lr_scheduler_type"],
            logging_steps=train_cfg["logging_steps"],
            save_steps=train_cfg["save_steps"],
            eval_strategy="steps",
            eval_steps=train_cfg["eval_steps"],
            fp16=train_cfg["fp16"],
            gradient_checkpointing=train_cfg["gradient_checkpointing"],
            report_to="none",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
        )

        # 5. Train
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=training_args,
            packing=False,
        )

        print("\n🚀 Starting student SFT training...")
        train_result = trainer.train()
        print(f"\n✅ Training complete. Loss: {train_result.training_loss:.4f}")

        # 6. Save adapter
        final_adapter_dir = Path(output_dir) / "final_adapter"
        model.save_pretrained(str(final_adapter_dir))
        tokenizer.save_pretrained(str(final_adapter_dir))
        print(f"💾 Saved student adapter to {final_adapter_dir}")

        # 7. Evaluate
        print("\n📊 Running evaluation...")
        eval_results = trainer.evaluate()
        metrics = {
            "stage": STAGE_NAME,
            "eval_loss": eval_results["eval_loss"],
            "train_loss": train_result.training_loss,
            "epochs": train_cfg["epochs"],
        }

        # 8. Upload + cleanup
        if s3_cfg["upload_checkpoint"]:
            print("\n☁️ Uploading to S3...")
            upload_checkpoint(
                checkpoint_dir=str(final_adapter_dir),
                stage_name=STAGE_NAME,
                s3_prefix=s3_cfg["prefix"],
                cleanup=s3_cfg["cleanup_after_upload"],
            )
            upload_metrics(metrics, STAGE_NAME)

        notify_stage_complete(STAGE_NAME, metrics)
        print("\n🎉 Stage B complete!")

    except Exception as e:
        notify_stage_error(STAGE_NAME, str(e))
        raise


def _validate_data(data_cfg: dict):
    """Validate data files."""
    import json
    for split in ["train_path", "eval_path"]:
        path = PROJECT_ROOT / data_cfg[split]
        if not path.exists():
            print(f"   ⚠️ {split}: {path} not found. Run 'python data/prepare.py' first.")
        else:
            with open(path, "r") as f:
                first_line = json.loads(f.readline())
            assert "messages" in first_line, f"Missing 'messages' field in {path}"
            print(f"   ✅ {split}: {path} valid")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage B: Student SFT Training")
    parser.add_argument("--dry-run", action="store_true", help="Validate config only")
    args = parser.parse_args()
    train(dry_run=args.dry_run)
