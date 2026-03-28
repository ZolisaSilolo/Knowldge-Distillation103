"""
dataset.py — HuggingFace Dataset loader with chat-template tokenization.

Handles both:
- SFT format (Stages C & B): standard chat messages → tokenized input/labels
- Distillation format (Stage A): includes teacher logits path reference
"""

import json
from pathlib import Path
from typing import Optional

from datasets import Dataset
from transformers import PreTrainedTokenizer


def load_sft_dataset(
    data_path: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 2048,
    split: str = "train",
) -> Dataset:
    """
    Load a JSONL dataset and tokenize for supervised fine-tuning.
    
    Each line in the JSONL should have a "messages" field with chat-format messages.
    
    Args:
        data_path: Path to the JSONL file.
        tokenizer: HuggingFace tokenizer with chat template support.
        max_length: Maximum sequence length.
        split: Dataset split identifier (for logging).
        
    Returns:
        Dataset: Tokenized HuggingFace Dataset ready for training.
    """
    # Load JSONL
    examples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    print(f"📂 Loaded {len(examples)} examples from {data_path} ({split})")

    # Tokenize using chat template
    def tokenize_fn(example):
        messages = example["messages"]

        # Apply chat template — produces the full conversation with special tokens
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        encodings = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )

        # For SFT, labels = input_ids with padding masked out
        labels = encodings["input_ids"].clone()
        labels[labels == tokenizer.pad_token_id] = -100
        encodings["labels"] = labels

        # Flatten from batch dim
        return {k: v.squeeze(0) for k, v in encodings.items()}

    dataset = Dataset.from_list(examples)
    tokenized = dataset.map(
        tokenize_fn,
        remove_columns=dataset.column_names,
        desc=f"Tokenizing {split}",
    )

    print(f"✅ Tokenized {len(tokenized)} examples (max_length={max_length})")
    return tokenized


def load_distillation_dataset(
    data_path: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 2048,
) -> Dataset:
    """
    Load a JSONL dataset for Stage A (KL Logit Distillation).
    
    Each example includes messages and optionally a teacher_logits_path
    pointing to pre-computed teacher logit tensors.
    
    Args:
        data_path: Path to the JSONL file.
        tokenizer: HuggingFace tokenizer.
        max_length: Maximum sequence length.
        
    Returns:
        Dataset: Tokenized dataset with teacher logit path metadata.
    """
    examples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    print(f"📂 Loaded {len(examples)} distillation examples from {data_path}")

    def tokenize_fn(example):
        messages = example["messages"]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        encodings = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )

        result = {k: v.squeeze(0) for k, v in encodings.items()}
        labels = result["input_ids"].clone()
        labels[labels == tokenizer.pad_token_id] = -100
        result["labels"] = labels

        # Preserve teacher logits path if available
        if "teacher_logits_path" in example and example["teacher_logits_path"]:
            result["teacher_logits_path"] = example["teacher_logits_path"]

        return result

    dataset = Dataset.from_list(examples)
    
    # Keep teacher_logits_path column if it exists
    remove_cols = [c for c in dataset.column_names if c not in ["teacher_logits_path"]]
    
    tokenized = dataset.map(
        tokenize_fn,
        remove_columns=remove_cols,
        desc="Tokenizing distillation data",
    )

    print(f"✅ Tokenized {len(tokenized)} distillation examples")
    return tokenized


def create_eval_texts(data_path: str, max_samples: int = 100) -> list[str]:
    """
    Load evaluation examples as raw text strings for perplexity computation.
    
    Args:
        data_path: Path to the eval JSONL file.
        max_samples: Maximum number of samples to load.
        
    Returns:
        list[str]: Raw text strings from evaluation examples.
    """
    texts = []
    with open(data_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            line = line.strip()
            if line:
                example = json.loads(line)
                # Concatenate all message contents
                text = " ".join(
                    msg["content"] for msg in example["messages"]
                )
                texts.append(text)
    return texts
