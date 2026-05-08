"""
Prefix Tuning — Llama 3.2 1B Instruct
Dataset : cybersecurity_lora_dataset.jsonl
Method  : Learnable virtual tokens prepended to every layer's KV cache.
          Base model fully frozen. ~0.1% trainable params.
VRAM    : ~6 GB
"""

import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments, DataCollatorForSeq2Seq,
)
from peft import PrefixTuningConfig, TaskType, get_peft_model
from hf_auth import login_huggingface, ensure_model

MODEL_ID   = "meta-llama/Llama-3.2-1B-Instruct"
DATA_PATH  = "cybersecurity_lora_dataset.jsonl"
OUTPUT_DIR = "./outputs/prefix_tuning"
MAX_LEN    = 512

PREFIX_CONFIG = PrefixTuningConfig(
    task_type          = TaskType.CAUSAL_LM,
    num_virtual_tokens = 20,
    prefix_projection  = True,
    encoder_hidden_size= 512,
)

TRAIN_ARGS = TrainingArguments(
    output_dir                  = OUTPUT_DIR,
    num_train_epochs            = 5,
    per_device_train_batch_size = 4,
    gradient_accumulation_steps = 4,
    learning_rate               = 3e-3,
    lr_scheduler_type           = "linear",
    warmup_steps                = 50,
    fp16                        = torch.cuda.is_available(),
    logging_steps               = 10,
    save_strategy               = "epoch",
    report_to                   = "none",
    optim                       = "adamw_torch",
)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def tokenize(sample, tokenizer):
    messages_full   = [{"role": "user",      "content": sample["user"]},
                       {"role": "assistant", "content": sample["assistant"]}]
    messages_prompt = [{"role": "user",      "content": sample["user"]}]

    full   = tokenizer.apply_chat_template(messages_full,   tokenize=False, add_generation_prompt=False)
    prompt = tokenizer.apply_chat_template(messages_prompt, tokenize=False, add_generation_prompt=True)

    enc        = tokenizer(full,   truncation=True, max_length=MAX_LEN, padding="max_length")
    prompt_len = len(tokenizer(prompt, truncation=True, max_length=MAX_LEN)["input_ids"])

    labels = enc["input_ids"].copy()
    labels[:prompt_len] = [-100] * prompt_len
    labels = [l if m else -100 for l, m in zip(labels, enc["attention_mask"])]
    enc["labels"] = labels
    return enc


def main():
    login_huggingface()
    ensure_model(MODEL_ID)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = Dataset.from_list(load_jsonl(DATA_PATH)).map(
        lambda x: tokenize(x, tokenizer), remove_columns=["user", "assistant"]
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model = get_peft_model(model, PREFIX_CONFIG)
    model.print_trainable_parameters()

    Trainer(
        model         = model,
        args          = TRAIN_ARGS,
        train_dataset = dataset,
        data_collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, label_pad_token_id=-100),
    ).train()

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Prefix tuning model saved → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
