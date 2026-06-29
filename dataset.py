"""
Dataset adapter + collator for Qwen2-VL fine-tuning.

Design goal: keep the training loop dataset-agnostic. Any HF dataset that has
an image column plus a question/answer (or caption) column can be plugged in by
editing the `data:` field mapping in config.yaml — no code changes needed.
"""

from PIL import Image
from datasets import load_dataset
from qwen_vl_utils import process_vision_info


def _to_pil(img):
    """HF datasets may give a PIL image, a dict with bytes, or a path."""
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, dict) and "bytes" in img and img["bytes"]:
        import io
        return Image.open(io.BytesIO(img["bytes"])).convert("RGB")
    if isinstance(img, str):
        return Image.open(img).convert("RGB")
    raise ValueError(f"Unsupported image type: {type(img)}")


def build_messages(example, cfg):
    """Turn one raw row into Qwen2-VL chat-format messages."""
    dcfg = cfg["data"]
    image = _to_pil(example[dcfg["image_field"]])
    question = str(example[dcfg["question_field"]])
    answer = str(example[dcfg["answer_field"]])

    user_content = [
        {"type": "image", "image": image},
        {"type": "text", "text": question},
    ]
    messages = [
        {"role": "system", "content": dcfg["system_prompt"]},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": answer},
    ]
    return messages


def load_hf_dataset(cfg):
    dcfg = cfg["data"]
    ds = load_dataset(dcfg["dataset_name"], split=dcfg["split"])
    if dcfg.get("max_samples"):
        ds = ds.select(range(min(dcfg["max_samples"], len(ds))))
    return ds


class QwenVLCollator:
    """
    Builds a training batch: applies the chat template, processes vision, tokenizes,
    and masks pad + image placeholder tokens in the labels.

    NOTE: this trains on the full sequence (prompt + answer), a common and effective
    simplification for instruction tuning. For answer-only loss, mask the prompt span
    per-sample before batching — easy to add later once the pipeline runs.
    """

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
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )

        labels = batch["input_ids"].clone()
        # Mask pad tokens
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        # Mask image pad/placeholder tokens so loss isn't computed on them
        for tok_name in ("image_token_id", "video_token_id"):
            tok_id = getattr(self.processor, tok_name, None)
            if tok_id is not None:
                labels[labels == tok_id] = -100
        batch["labels"] = labels
        return batch
