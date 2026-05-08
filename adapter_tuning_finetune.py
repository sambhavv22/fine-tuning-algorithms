"""
Adapter Tuning for Llama 3.2 1B Instruct
Inserts small bottleneck MLP adapters into every transformer layer.
Only adapter weights are trained; the base model is frozen.
Uses PEFT's IA3 method (Infused Adapter by Inhibiting and Amplifying) —
the most lightweight adapter variant, rescaling attention keys/values and FFN.
Dataset: cybersecurity_lora_dataset.jsonl

Note: For classic Houlsby-style bottleneck adapters use peft.AdaptionPromptConfig
      or adapter-transformers library. IA3 is natively supported in PEFT and
      achieves comparable results with fewer parameters.
"""

import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import IA3Config, TaskType, get_peft_model

from hf_auth import login_huggingface, ensure_model

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID   = "meta-llama/Llama-3.2-1B-Instruct"
DATA_PATH  = "cybersecurity_lora_dataset.jsonl"
OUTPUT_DIR = "./outputs/adapter_tuning"
MAX_LEN    = 512

IA3_CONFIG = IA3Config(
    task_type      = TaskType.CAUSAL_LM,
    # Modules to insert IA3 scaling vectors into
    target_modules = ["k_proj", "v_proj", "down_proj"],
    feedforward_modules = ["down_proj"],   # rescale FFN outputs
)

TRAIN_ARGS = TrainingArguments(
    output_dir              = OUTPUT_DIR,
    num_train_epochs        = 5,
    per_device_train_batch_size = 8,       # IA3 is so lightweight we can use large batches
    gradient_accumulation_steps = 2,
    learning_rate           = 3e-3,
    lr_scheduler_type       = "cosine",
    warmup_ratio            = 0.05,
    fp16                    = torch.cuda.is_available(),
    logging_steps           = 10,
    save_strategy           = "epoch",
    report_to               = "none",
    optim                   = "adamw_torch",
)

# ── Data ──────────────────────────────────────────────────────────────────────
def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]

def format_chat(sample, tokenizer):
    messages = [
        {"role": "user",      "content": sample["user"]},
        {"role": "assistant", "content": sample["assistant"]},
    ]
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": sample["user"]}],
        tokenize=False, add_generation_prompt=True
    )
    enc_full   = tokenizer(full_text,   truncation=True, max_length=MAX_LEN, padding="max_length")
    enc_prompt = tokenizer(prompt_text, truncation=True, max_length=MAX_LEN)
    prompt_len = len(enc_prompt["input_ids"])

    labels = enc_full["input_ids"].copy()
    labels[:prompt_len] = [-100] * prompt_len
    labels = [l if m else -100 for l, m in zip(labels, enc_full["attention_mask"])]
    enc_full["labels"] = labels
    return enc_full

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
# ── Auth / model download ─────────────────────────────────────────────────────
login_huggingface()
ensure_model(MODEL_ID)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    raw     = load_jsonl(DATA_PATH)
    dataset = Dataset.from_list(raw).map(
        lambda x: format_chat(x, tokenizer),
        remove_columns=["user", "assistant"],
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map  = "auto",
        trust_remote_code = True,
    )
    model = get_peft_model(model, IA3_CONFIG)
    model.print_trainable_parameters()

    trainer = Trainer(
        model         = model,
        args          = TRAIN_ARGS,
        train_dataset = dataset,
        data_collator = DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, label_pad_token_id=-100
        ),
    )
    trainer.train()
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"IA3 adapter saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
