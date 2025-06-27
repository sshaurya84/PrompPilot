import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, DataCollatorForLanguageModeling, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from transformers.utils.quantization_config import BitsAndBytesConfig

# Configuration
MODEL_NAME = "EleutherAI/gpt-neo-1.3B"
DATASET_PATH = "./lora-finetune/dataset/train.json"
OUTPUT_DIR = "./lora-model"
USE_8BIT = False

dataset = load_dataset("json", data_files={"train": DATASET_PATH})["train"]

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token  # Set padding token for causal models

# Tokenize each example
def tokenize(example):
    tokenized = tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=512,
    )
    # Set labels = input_ids (standard for causal LM training)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

# Apply tokenization
tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)


model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16,
)
model.config.use_cache = False  
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["attn.attention.out_proj", "mlp.c_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="epoch",
    bf16=True if torch.cuda.is_available() else False,
    fp16=not torch.cuda.is_bf16_supported(),
    remove_unused_columns=False,
    save_total_limit=2,
    report_to="none",
)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # MLM = False because GPT-style models use causal LM
)
# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# Train
trainer.train()

# Save adapter
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
