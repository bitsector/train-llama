import pandas as pd
import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, AutoPeftModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
# from pytrl import SFTTrainer
from trl import SFTTrainer, SFTConfig

# Model from Hugging Face hub
base_model = "NousResearch/Llama-2-7b-chat-hf"

# New instruction dataset
dolly_15K= "databricks/databricks-dolly-15k"

# Fine-tuned model
new_model = "llama-2-7b-chat-dolly"


# Download the dataset
dataset = load_dataset(dolly_15K, split="train")


print(f'Number of prompts: {len(dataset)}')
print(f'Column names are: {dataset.column_names}')

# def create_prompt(row):
#     prompt = f"Instruction: {row['instruction']}\\nContext: {row['context']}\\nResponse: {row['response']}"
#     return prompt


# print("dataset type: ",type(dataset))
# dataset['text'] = dataset.apply(create_prompt, axis=1)
# data = Dataset.from_pandas(dataset)


def create_prompt(example):
    return {
        "text": f"Instruction: {example['instruction']}\nContext: {example['context']}\nResponse: {example['response']}"
    }

dataset = dataset.map(create_prompt)
data = dataset.to_pandas()

# Get the type
compute_dtype = getattr(torch, "float16")


# BitsAndBytesConfig int-4 config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype
   
)


# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


peft_config = LoraConfig(r=32,
                        lora_alpha=64,
                        lora_dropout=0.05,
                        bias="none",
                        task_type="CAUSAL_LM"
                      )


# Define the training arguments. For full list of arguments, check
#<https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments>
args = SFTConfig(
    output_dir='llama-dolly-7b',
    warmup_steps=1,
    num_train_epochs=10,
    per_device_train_batch_size=2,
    save_strategy="epoch",
    logging_steps=50,
    optim="paged_adamw_32bit",
    learning_rate=2.5e-5,
    fp16=True,
    max_steps=500,
    save_steps=50,
    do_eval=False,
    packing=True  # Add this line to enable packing
)

# Create the trainer
trainer = SFTTrainer(
    model=base_model,
    train_dataset=dataset,
    peft_config=peft_config,
    args=args
)
