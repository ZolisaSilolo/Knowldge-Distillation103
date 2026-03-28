"""
Stage A — KL Logit Distillation (Raw PyTorch)

This is the core knowledge distillation stage. The student model learns to match
the teacher's soft probability distribution via KL divergence loss, combined
with standard cross-entropy loss on hard labels.

** VRAM SAFETY **
Unlike Stages C/B (protected by Unsloth), this stage uses raw PyTorch.
On a free T4 (15GB VRAM), we MUST enable:
- torch.utils.checkpoint (gradient checkpointing)
- 8-bit AdamW optimizer via bitsandbytes
- Batch size = 1 with high gradient accumulation
- Periodic torch.cuda.empty_cache()

Usage:
    python stage_a/distill.py
    python stage_a/distill.py --dry-run
"""

import argparse
import sys
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import load_config
from utils.checkpoint import upload_checkpoint, upload_metrics
from utils.notify import notify_stage_start, notify_stage_complete, notify_stage_error


STAGE_NAME = "stage_a"


class DistillationLoss(nn.Module):
    """
    Combined KL Divergence + Cross-Entropy loss for knowledge distillation.
    
    Loss = alpha * KL(student_soft || teacher_soft) + (1 - alpha) * CE(student, labels)
    
    The KL term transfers "dark knowledge" from the teacher's soft probability
    distribution, while the CE term keeps the student grounded on correct labels.
    """

    def __init__(self, temperature: float = 2.0, alpha: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute distillation loss.
        
        Args:
            student_logits: Student model logits [batch, seq_len, vocab_size]
            teacher_logits: Teacher model logits [batch, seq_len, vocab_size]
            labels: Ground truth token IDs [batch, seq_len]
            
        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Temperature-scaled soft targets
        T = self.temperature
        student_soft = F.log_softmax(student_logits / T, dim=-1)
        teacher_soft = F.softmax(teacher_logits / T, dim=-1)

        # KL divergence on soft distributions
        # Multiply by T^2 to maintain gradient magnitudes (Hinton et al., 2015)
        kl_loss = F.kl_div(
            student_soft,
            teacher_soft,
            reduction="batchmean",
        ) * (T ** 2)

        # Standard cross-entropy on hard labels
        ce_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )

        # Combined loss
        loss = self.alpha * kl_loss + (1 - self.alpha) * ce_loss
        return loss


def distill(dry_run: bool = False):
    """
    Run Stage A: KL Logit Distillation.
    
    Args:
        dry_run: If True, validate config only.
    """
    config = load_config(f"{STAGE_NAME}/config.yaml")
    model_cfg = config["model"]
    distill_cfg = config["distillation"]
    vram_cfg = config["vram_safety"]
    data_cfg = config["data"]
    output_cfg = config["output"]
    s3_cfg = config["s3"]

    print("=" * 60)
    print(f"ClinIQ — {STAGE_NAME.upper()}: KL Logit Distillation")
    print(f"Teacher: {model_cfg['teacher']}")
    print(f"Student: {model_cfg['student']}")
    print(f"Temperature: {distill_cfg['temperature']}, Alpha: {distill_cfg['alpha']}")
    print(f"Batch: {distill_cfg['per_device_batch_size']} × {distill_cfg['gradient_accumulation_steps']} accum")
    print(f"VRAM Safety: grad_ckpt={vram_cfg['gradient_checkpointing']}, 8bit_optim={vram_cfg['use_8bit_optimizer']}")
    print("=" * 60)

    if dry_run:
        print("\n🔍 DRY RUN — validating config...")
        assert distill_cfg["per_device_batch_size"] <= 2, "Batch size must be ≤2 for T4 VRAM safety"
        assert vram_cfg["gradient_checkpointing"], "Gradient checkpointing MUST be enabled"
        assert vram_cfg["use_8bit_optimizer"], "8-bit optimizer MUST be enabled for VRAM safety"
        print("✅ VRAM safety checks passed.")
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from data.dataset import load_distillation_dataset

    notify_stage_start(STAGE_NAME, distill_cfg)

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n🖥️ Device: {device}")
        if device.type == "cuda":
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

        # ===== 1. Load Teacher (frozen, 4-bit for T4 VRAM safety) =====
        print("\n📦 Loading teacher model (frozen, 4-bit)...")
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        teacher_model = AutoModelForCausalLM.from_pretrained(
            model_cfg["teacher"],
            quantization_config=bnb_config,
            device_map="auto",
        )
        # Only load adapter if specified — GGUF repos have no safetensors adapter
        teacher_adapter = model_cfg.get("teacher_adapter")
        if teacher_adapter:
            teacher_model = PeftModel.from_pretrained(
                teacher_model, teacher_adapter, is_trainable=False,
            )
            teacher_model = teacher_model.merge_and_unload()
            print("   ✅ Teacher adapter merged.")
        else:
            print("   ℹ️  No adapter — using base teacher model directly.")

        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False
        print("   ✅ Teacher frozen (no gradients).")

        # ===== 2. Load Student (trainable) =====
        print("\n📦 Loading student model (trainable)...")
        student_model = AutoModelForCausalLM.from_pretrained(
            model_cfg["student"],
            torch_dtype=torch.float16,
            device_map="auto",
        )
        student_adapter_path = PROJECT_ROOT / model_cfg["student_adapter"]
        if student_adapter_path.exists():
            student_model = PeftModel.from_pretrained(
                student_model, str(student_adapter_path), is_trainable=True
            )
            print("   ✅ Student adapter loaded (trainable).")

        # Enable gradient checkpointing for VRAM safety
        if vram_cfg["gradient_checkpointing"]:
            student_model.gradient_checkpointing_enable()
            print("   ✅ Gradient checkpointing enabled.")

        # ===== 3. Load Tokenizer & Data =====
        tokenizer = AutoTokenizer.from_pretrained(model_cfg["student"])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print("\n📂 Loading distillation data...")
        train_dataset = load_distillation_dataset(
            data_path=str(PROJECT_ROOT / data_cfg["train_path"]),
            tokenizer=tokenizer,
            max_length=data_cfg["max_length"],
        )
        train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

        train_loader = DataLoader(
            train_dataset,
            batch_size=distill_cfg["per_device_batch_size"],
            shuffle=True,
            drop_last=True,
        )

        # ===== 4. Setup Optimizer (8-bit for VRAM safety) =====
        grad_accum_steps = distill_cfg["gradient_accumulation_steps"]
        cache_freq = vram_cfg["empty_cache_frequency"]

        if vram_cfg["use_8bit_optimizer"]:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(
                student_model.parameters(),
                lr=distill_cfg["learning_rate"],
                weight_decay=distill_cfg["weight_decay"],
            )
            print("   ✅ Using 8-bit AdamW optimizer (bitsandbytes).")
        else:
            optimizer = torch.optim.AdamW(
                student_model.parameters(),
                lr=distill_cfg["learning_rate"],
                weight_decay=distill_cfg["weight_decay"],
            )

        # ===== 5. Learning rate scheduler with linear warmup =====
        total_steps = (len(train_loader) // grad_accum_steps) * distill_cfg["epochs"]
        warmup_steps = distill_cfg["warmup_steps"]

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # ===== 6. Distillation Loss =====
        criterion = DistillationLoss(
            temperature=distill_cfg["temperature"],
            alpha=distill_cfg["alpha"],
        )

        # ===== 7. Training Loop =====
        print(f"\n🚀 Starting distillation ({distill_cfg['epochs']} epochs)...")
        scaler = torch.amp.GradScaler("cuda") if vram_cfg["mixed_precision"] else None
        global_step = 0

        for epoch in range(distill_cfg["epochs"]):
            student_model.train()
            epoch_loss = 0.0
            num_batches = 0

            for batch_idx, batch in enumerate(train_loader):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                # Forward pass with mixed precision
                with torch.amp.autocast("cuda", enabled=vram_cfg["mixed_precision"]):
                    # Teacher forward (no gradients)
                    with torch.no_grad():
                        teacher_outputs = teacher_model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                        )
                        teacher_logits = teacher_outputs.logits

                    # Student forward
                    student_outputs = student_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
                    student_logits = student_outputs.logits

                    # Handle vocab size mismatch between teacher and student
                    min_vocab = min(student_logits.size(-1), teacher_logits.size(-1))
                    loss = criterion(
                        student_logits[..., :min_vocab],
                        teacher_logits[..., :min_vocab],
                        labels,
                    )
                    loss = loss / grad_accum_steps

                # Backward pass
                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                # Gradient accumulation step
                if (batch_idx + 1) % grad_accum_steps == 0:
                    if scaler:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            student_model.parameters(),
                            distill_cfg["max_grad_norm"],
                        )
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            student_model.parameters(),
                            distill_cfg["max_grad_norm"],
                        )
                        optimizer.step()

                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1

                epoch_loss += loss.item() * grad_accum_steps
                num_batches += 1

                # Periodic VRAM cleanup
                if batch_idx % cache_freq == 0:
                    torch.cuda.empty_cache()

                # Logging
                if batch_idx % 10 == 0:
                    current_lr = optimizer.param_groups[0]["lr"]
                    vram_used = torch.cuda.memory_allocated() / 1e9 if device.type == "cuda" else 0
                    print(
                        f"  Epoch {epoch+1}/{distill_cfg['epochs']} | "
                        f"Batch {batch_idx}/{len(train_loader)} | "
                        f"Loss: {loss.item() * grad_accum_steps:.4f} | "
                        f"LR: {current_lr:.2e} | "
                        f"VRAM: {vram_used:.1f}GB"
                    )

            avg_loss = epoch_loss / max(num_batches, 1)
            print(f"\n📊 Epoch {epoch+1} — Avg Loss: {avg_loss:.4f}")

        # ===== 8. Eval loop (required for Lambda compare_models) =====
        print("\n📊 Running eval loop...")
        student_model.eval()
        eval_dataset = load_distillation_dataset(
            data_path=str(PROJECT_ROOT / data_cfg["eval_path"]),
            tokenizer=tokenizer,
            max_length=data_cfg["max_length"],
        )
        eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        eval_loader = DataLoader(eval_dataset, batch_size=distill_cfg["per_device_batch_size"], shuffle=False)
        eval_loss_total = 0.0
        eval_batches = 0
        with torch.no_grad():
            for batch in eval_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                with torch.amp.autocast("cuda", enabled=vram_cfg["mixed_precision"]):
                    t_out = teacher_model(input_ids=input_ids, attention_mask=attention_mask)
                    s_out = student_model(input_ids=input_ids, attention_mask=attention_mask)
                    min_vocab = min(s_out.logits.size(-1), t_out.logits.size(-1))
                    loss = criterion(s_out.logits[..., :min_vocab], t_out.logits[..., :min_vocab], labels)
                eval_loss_total += loss.item()
                eval_batches += 1
        eval_loss = eval_loss_total / max(eval_batches, 1)
        print(f"   Eval Loss: {eval_loss:.4f}")

        # ===== 9. Save final distilled student =====
        output_dir = Path(PROJECT_ROOT / output_cfg["dir"])
        final_dir = output_dir / "final_adapter"
        final_dir.mkdir(parents=True, exist_ok=True)

        student_model.save_pretrained(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))
        print(f"\n💾 Saved distilled student to {final_dir}")

        # ===== 9. Upload to S3 =====
        metrics = {
            "stage": STAGE_NAME,
            "eval_loss": eval_loss,
            "train_loss": avg_loss,
            "temperature": distill_cfg["temperature"],
            "alpha": distill_cfg["alpha"],
            "epochs": distill_cfg["epochs"],
        }

        if s3_cfg["upload_checkpoint"]:
            print("\n☁️ Uploading to S3...")
            upload_checkpoint(
                checkpoint_dir=str(final_dir),
                stage_name=STAGE_NAME,
                s3_prefix=s3_cfg["prefix"],
                cleanup=s3_cfg["cleanup_after_upload"],
            )
            upload_metrics(metrics, STAGE_NAME)

        notify_stage_complete(STAGE_NAME, metrics)
        print("\n🎉 Stage A (Distillation) complete!")

    except Exception as e:
        notify_stage_error(STAGE_NAME, str(e))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage A: KL Logit Distillation")
    parser.add_argument("--dry-run", action="store_true", help="Validate config only")
    args = parser.parse_args()
    distill(dry_run=args.dry_run)
