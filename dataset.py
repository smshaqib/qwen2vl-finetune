"""
Dataset adapter + collator for Qwen2-VL fine-tuning.

Supports two sources (chosen by `data.source` in config):
  * local_csv : a CSV of {image_name, label} + an image folder  (your meme dataset)
  * hf        : any Hugging Face dataset with image/question/answer columns

The task here is meme classification: the model sees the meme image and must
reply with exactly one category. The collator masks pad + image tokens so loss
is computed on the text answer.
"""

import os
import csv
import random

from PIL import Image
from datasets import Dataset, load_dataset
from qwen_vl_utils import process_vision_info


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _load_local_csv(cfg):
    d = cfg["data"]
    rows = []
    with open(d["csv_path"], encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            img_path = os.path.join(d["images_dir"], row[d["image_field"]])
            rows.append({"image_path": img_path, "answer": str(row[d["answer_field"]])})
    random.Random(cfg["train"]["seed"]).shuffle(rows)
    if d.get("max_samples"):
        rows = rows[: d["max_samples"]]
    return Dataset.from_list(rows)


def _load_hf(cfg):
    d = cfg["data"]
    ds = load_dataset(d["dataset_name"], split=d["split"])
    if d.get("max_samples"):
        ds = ds.select(range(min(d["max_samples"], len(ds))))
    return ds


def load_dataset_split(cfg):
    """Return a HF Dataset for training. For local_csv, optionally hold out a
    validation slice (data.val_fraction) returned as the second element."""
    d = cfg["data"]
    if d.get("source", "hf") == "local_csv":
        ds = _load_local_csv(cfg)
        vf = d.get("val_fraction", 0.0)
        if vf and vf > 0:
            n_val = max(1, int(len(ds) * vf))
            return ds.select(range(n_val, len(ds))), ds.select(range(n_val))
        return ds, None
    return _load_hf(cfg), None


# --------------------------------------------------------------------------- #
# Message building
# --------------------------------------------------------------------------- #
def _to_pil(img):
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, str):
        return Image.open(img).convert("RGB")
    if isinstance(img, dict) and img.get("bytes"):
        import io
        return Image.open(io.BytesIO(img["bytes"])).convert("RGB")
    raise ValueError(f"Unsupported image type: {type(img)}")


def build_messages(example, cfg, with_answer=True):
    """One raw row -> Qwen2-VL chat messages. `question` is a fixed instruction
    (classification), and the system prompt enumerates the allowed labels."""
    d = cfg["data"]
    if d.get("source", "hf") == "local_csv":
        image = _to_pil(example["image_path"])
        answer = example["answer"]
    else:
        image = _to_pil(example[d["image_field"]])
        answer = str(example[d["answer_field"]])

    question = d.get("question") or str(example.get(d.get("question_field", ""), ""))
    messages = [
        {"role": "system", "content": d["system_prompt"]},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question},
        ]},
    ]
    if with_answer:
        messages.append({"role": "assistant", "content": answer})
    return messages


class QwenVLCollator:
    """Applies the chat template, processes vision, tokenizes, and builds labels
    (pad + image placeholder tokens masked to -100).

    NOTE: trains on the full sequence (prompt + answer). Since the answer here is
    a single short category word, this is fine and stable."""

    def __init__(self, processor, cfg):
        self.processor = processor
        self.cfg = cfg
        self.max_len = cfg["train"]["max_seq_length"]

    def __call__(self, examples):
        messages_list = [build_messages(ex, self.cfg) for ex in examples]
        texts = [
            self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
            for m in messages_list
        ]
        image_inputs, video_inputs = process_vision_info(messages_list)
        batch = self.processor(
            text=texts, images=image_inputs, videos=video_inputs,
            padding=True, truncation=True, max_length=self.max_len,
            return_tensors="pt",
        )
        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        for tok_name in ("image_token_id", "video_token_id"):
            tok_id = getattr(self.processor, tok_name, None)
            if tok_id is not None:
                labels[labels == tok_id] = -100
        batch["labels"] = labels
        return batch
