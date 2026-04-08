import os
import time
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

# =========================
# CONFIG (ENV FIRST)
# =========================
BASE_MODEL = os.getenv("BASE_MODEL", "/models/finetune/base/Qwen3.5-9B")

DATA_DIR = os.getenv("DATA_DIR", "/models/finetune/datasets")
TRAIN_PATH = os.getenv("TRAIN_PATH", f"{DATA_DIR}/train.jsonl")
VAL_PATH = os.getenv("VAL_PATH", f"{DATA_DIR}/val.jsonl")

DATASET_NAME = os.getenv("DATASET_NAME", "LLMs/Alpaca-ShareGPT")
MAX_SAMPLES = int(os.getenv("MAX_SAMPLES", "50000"))
VAL_SPLIT = float(os.getenv("VAL_SPLIT", "0.1"))

OUTPUT_BASE = os.getenv("OUTPUT_BASE", "/models/finetune")
EXP_NAME = os.getenv("EXP_NAME", f"exp-{int(time.time())}")
EXP_DIR = f"{OUTPUT_BASE}/experiments/{EXP_NAME}"
FINAL_MODEL_DIR = f"{EXP_DIR}/final_model"

# LoRA params
LORA_R = int(os.getenv("LORA_R", "16"))
LORA_ALPHA = int(os.getenv("LORA_ALPHA", "32"))

# Training params
EPOCHS = int(os.getenv("EPOCHS", "1"))
LR = float(os.getenv("LR", "2e-4"))

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(EXP_DIR, exist_ok=True)

print("==== CONFIG ====")
for k, v in {
    "BASE_MODEL": BASE_MODEL,
    "TRAIN_PATH": TRAIN_PATH,
    "VAL_PATH": VAL_PATH,
    "DATASET_NAME": DATASET_NAME,
    "MAX_SAMPLES": MAX_SAMPLES,
    "VAL_SPLIT": VAL_SPLIT,
    "EXP_DIR": EXP_DIR
}.items():
    print(f"{k}: {v}")

# =========================
# STEP 1: DATA DOWNLOAD
# =========================
if not os.path.exists(TRAIN_PATH):
    print("train.jsonl not found → downloading dataset...")

    dataset = load_dataset(DATASET_NAME)

    dataset = dataset["train"].shuffle(seed=42).select(range(MAX_SAMPLES))
    dataset.to_json(TRAIN_PATH)

    print(f"Saved dataset → {TRAIN_PATH}")
else:
    print("train.jsonl exists → skip download")

# =========================
# STEP 2: TRAIN / VAL SPLIT
# =========================
if not os.path.exists(VAL_PATH):
    print("val.jsonl not found → splitting dataset...")

    dataset = load_dataset("json", data_files=TRAIN_PATH)["train"]
    split = dataset.train_test_split(test_size=VAL_SPLIT, seed=42)

    split["train"].to_json(TRAIN_PATH)
    split["test"].to_json(VAL_PATH)

    print("Split completed")
else:
    print("val.jsonl exists → skip split")

# =========================
# STEP 3: LOAD DATA
# =========================
train_dataset = load_dataset("json", data_files=TRAIN_PATH)["train"]
val_dataset = load_dataset("json", data_files=VAL_PATH)["train"]

# =========================
# STEP 4: MODEL LOAD
# =========================
print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    dtype=torch.bfloat16
)

# CRITICAL FIXES FOR OOM:
# model.enable_input_require_grads()  # Required for LoRA + gradient checkpointing
# model.gradient_checkpointing_enable()  # Trades compute for memory (~50% savings)

# Add this - important for Qwen models
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# =========================
# STEP 5: LoRA CONFIG
# =========================
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# =========================
# STEP 6: TOKENIZATION (FIXED)
# =========================
def format_alpaca_prompt(example):
    """Format Alpaca-style: Instruction + Input + Output"""
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")
    
    # If input exists and is not empty, include it
    if input_text and str(input_text).strip():
        text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
    else:
        # Simple format when input is empty (as in your examples)
        text = f"{instruction}\n\n{output}"
    
    return text

def tokenize(example):
    text = format_alpaca_prompt(example)
    
    result = tokenizer(
        text,
        truncation=True,
        max_length=2048,
        padding="max_length"
    )
    
    # CRITICAL: Add labels for Causal LM training
    result["labels"] = result["input_ids"].copy()
    return result

# Remove batched=True to avoid the list concatenation error
train_dataset = train_dataset.map(
    tokenize, 
    remove_columns=train_dataset.column_names
)
val_dataset = val_dataset.map(
    tokenize, 
    remove_columns=val_dataset.column_names
)

# Set format for PyTorch tensors
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# =========================
# STEP 7: TRAINING
# =========================
# training_args = TrainingArguments(
#     output_dir=EXP_DIR,
#     per_device_train_batch_size=1,
#     gradient_accumulation_steps=8,
#     num_train_epochs=EPOCHS,
#     logging_steps=10,
#     save_steps=100,
#     save_total_limit=2,
#     eval_strategy="steps",
#     eval_steps=100,
#     bf16=True,
#     learning_rate=LR,
#     report_to="none"
# )

training_args = TrainingArguments(
    output_dir=EXP_DIR,
    per_device_train_batch_size=8,        # Increase from 1 to 8 (or even 16)
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=1,        # Reduce or keep at 2
    num_train_epochs=EPOCHS,
    logging_steps=10,
    save_steps=1000,
    save_total_limit=2,
    eval_strategy="steps",
    eval_steps=500,
    bf16=True,
    learning_rate=LR,
    report_to="none",
    remove_unused_columns=False,
    warmup_steps=100,                     # Changed from warmup_ratio (fixes deprecation warning)
    lr_scheduler_type="cosine",
    seed=42,
    dataloader_num_workers=4,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

print("Starting training...")
trainer.train()

# =========================
# STEP 8: SAVE MODEL
# =========================
print("Saving model...")

model.save_pretrained(FINAL_MODEL_DIR)
tokenizer.save_pretrained(FINAL_MODEL_DIR)

# Stable symlink
LATEST_LINK = f"{OUTPUT_BASE}/output/qwen9b-ft/latest"
os.makedirs(os.path.dirname(LATEST_LINK), exist_ok=True)

if os.path.islink(LATEST_LINK) or os.path.exists(LATEST_LINK):
    os.remove(LATEST_LINK)

os.symlink(EXP_DIR, LATEST_LINK)

print(f"Training complete → {FINAL_MODEL_DIR}")
