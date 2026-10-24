#!/usr/bin/env python3
"""
LinuxMate - LoRA Fine-Tuning Script for Qwen2.5-1.5B-Instruct
Hardware target: NVIDIA RTX 5060 Laptop GPU (8GB VRAM)
Dataset: train.jsonl (500 items, 60% commands : 40% OS theory)
"""
import os
import sys
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "train.jsonl")
OUTPUT_DIR = os.path.join(BASE_DIR, "linuxmate-lora")
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

def main():
    print("=" * 60)
    print(" LinuxMate | LoRA Fine-Tuning on RTX 5060 GPU")
    print(f" Base Model: {MODEL_ID}")
    print(f" Dataset   : {DATA_FILE}")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("[!] CUDA is not available. Fine-tuning requires an NVIDIA GPU.")
        sys.exit(1)

    device_name = torch.cuda.get_device_name(0)
    vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
    print(f"Detected GPU: {device_name} ({vram_gb} GB VRAM)")

    # 1. Load Tokenizer
    print("\n[1/4] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Prepare Dataset in ChatML format
    print("\n[2/4] Formatting dataset for instruction tuning...")
    raw_dataset = load_dataset("json", data_files=DATA_FILE, split="train")

    def format_chat(sample):
        system_prompt = (
            "You are LinuxMate, a specialized Linux command and Operating Systems lab assistant. "
            "For Linux command requests, reply strictly in the format:\n"
            "Command: <command>\nMeaning: <explanation>\nWarning: <only if destructive>\n"
            "For OS theory questions, provide concise, accurate viva-ready explanations."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": sample["instruction"]},
            {"role": "assistant", "content": sample["output"]},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        return {"text": text}

    dataset = raw_dataset.map(format_chat)
    print(f"Total formatted training examples: {len(dataset)}")

    # 3. Load Model in bfloat16 / float16
    print("\n[3/4] Loading base model into GPU memory...")
    # RTX 5060 supports bfloat16 natively
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True
    )

    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 4. Training configuration
    print("\n[4/4] Starting training run...")
    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        dataset_text_field="text",
        max_length=512,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,   # effective batch size = 8
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=25,                 # ~5% of ~500 steps
        logging_steps=10,
        fp16=(dtype == torch.float16),
        bf16=(dtype == torch.bfloat16),
        save_strategy="no",
        report_to="none"
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=sft_config,
    )

    trainer.train()
    print("\n[OK] Training completed! Saving LoRA adapter to:", OUTPUT_DIR)
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("[OK] All saved successfully! Ready to deploy.")

if __name__ == "__main__":
    main()
