"""
QLoRA fine-tuning for Qwen2-VL on a Kaggle GPU.

Run:   python train.py --config config.yaml

Designed for Kaggle constraints:
  * 4-bit base + LoRA adapters -> fits a single 16 GB T4
  * checkpoints to /kaggle/working and auto-resumes (sessions can drop at 9h)
  * fp16 (T4 has no bf16)
"""

import argparse
import os

import torch
import yaml
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from dataset import QwenVLCollator, load_hf_dataset


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_last_checkpoint(output_dir):
    if not os.path.isdir(output_dir):
        return None
    ckpts = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if not ckpts:
        return None
    ckpts.sort(key=lambda d: int(d.split("-")[-1]))
    return os.path.join(output_dir, ckpts[-1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    mcfg, tcfg = cfg["model"], cfg["train"]
    torch.manual_seed(tcfg["seed"])

    # ---- Processor (handles text + vision tokenization) -------------------
    processor = AutoProcessor.from_pretrained(
        mcfg["name"],
        min_pixels=mcfg["min_pixels"],
        max_pixels=mcfg["max_pixels"],
    )

    # ---- 4-bit base model -------------------------------------------------
    quant_cfg = None
    if mcfg["load_in_4bit"]:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        mcfg["name"],
        quantization_config=quant_cfg,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=tcfg["gradient_checkpointing"]
    )

    # ---- LoRA adapters (vision tower stays frozen) ------------------------
    lcfg = cfg["lora"]
    lora = LoraConfig(
        r=lcfg["r"],
        lora_alpha=lcfg["alpha"],
        lora_dropout=lcfg["dropout"],
        target_modules=lcfg["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    if tcfg["gradient_checkpointing"]:
        model.config.use_cache = False

    # ---- Data -------------------------------------------------------------
    train_ds = load_hf_dataset(cfg)
    collator = QwenVLCollator(processor, cfg)

    # ---- Trainer ----------------------------------------------------------
    targs = TrainingArguments(
        output_dir=tcfg["output_dir"],
        num_train_epochs=tcfg["num_epochs"],
        per_device_train_batch_size=tcfg["per_device_batch_size"],
        gradient_accumulation_steps=tcfg["grad_accum_steps"],
        learning_rate=tcfg["learning_rate"],
        warmup_ratio=tcfg["warmup_ratio"],
        weight_decay=tcfg["weight_decay"],
        logging_steps=tcfg["logging_steps"],
        save_steps=tcfg["save_steps"],
        save_total_limit=tcfg["save_total_limit"],
        bf16=tcfg["bf16"],
        fp16=tcfg["fp16"],
        gradient_checkpointing=tcfg["gradient_checkpointing"],
        report_to="none",
        remove_unused_columns=False,   # we hand-build batches in the collator
        dataloader_pin_memory=False,
        seed=tcfg["seed"],
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        data_collator=collator,
    )

    resume = None
    if tcfg.get("resume"):
        resume = find_last_checkpoint(tcfg["output_dir"])
        if resume:
            print(f"[resume] continuing from {resume}")

    trainer.train(resume_from_checkpoint=resume)

    # Save final LoRA adapter + processor
    trainer.save_model(tcfg["output_dir"])
    processor.save_pretrained(tcfg["output_dir"])
    print(f"[done] adapter saved to {tcfg['output_dir']}")


if __name__ == "__main__":
    main()
