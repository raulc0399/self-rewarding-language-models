# from https://www.oxen.ai/blog/how-to-run-llama-2-on-cpu-after-fine-tuning-with-lora

from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, TrainingArguments
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training,
)
from trl import SFTTrainer

import sys, os

if len(sys.argv) != 3:
    print("Usage: python 01_and_03_sft.py <dataset.json> <results_dir>")
    exit()

# use 00_ift.jsonl for step 1
# use 02.1_ift_and_eft.jsonl for step 3
dataset_file = sys.argv[1]
output_dir = sys.argv[2]

# load the training dataset
dataset = load_dataset("json", data_files={'train': dataset_file})
dataset = dataset['train'].shuffle(seed=42)

base_model_name = "mistralai/Mistral-7B-v0.1"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

device_map = "auto"

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    device_map=device_map,
    trust_remote_code=True,
)
base_model.config.use_cache = False

# from the LoRA paper:
# "In the Transformer architecture, there are four weight matrices in
# the self-attention module (Wq, Wk, Wv, Wo) and two in the MLP module.
# We limit our study to only adapting the attention weights for downstream tasks and freeze the MLP modules"
# print(base_model)

# More info: https://github.com/huggingface/transformers/pull/24906
base_model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# from instructions_formatting_function of trl\extras\dataset_formatting.py
converted_sample = [
    {"role": "user", "content": "prompt"},
    {"role": "assistant", "content": "completion"},
]
print(tokenizer.apply_chat_template(converted_sample, tokenize=False))

def dataset_stats(dataset, tokenizer):
    print(dataset)

    field_name = "prompt_response"
    
    lengths = [len(entry[field_name]) for entry in dataset]
    max_length = max(lengths)

    # Calculate the average length
    average_length = sum(lengths) / len(lengths)

    # Find the 60th percentile length
    lengths_sorted = sorted(lengths)
    index_60_percentile = int(0.6 * len(lengths_sorted))  # Calculate the index for the 60th percentile
    length_60_percentile = lengths_sorted[index_60_percentile]

    print(f"Maximum length of {field_name} entries: {max_length}")
    print(f"Average length of {field_name} entries: {average_length}")
    print(f"Length longer than 60% of {field_name} entries: {length_60_percentile}")

    # Tokenize the text and count the number of tokens for each entry
    token_counts = [len(tokenizer.encode(entry[field_name])) for entry in dataset]
    max_tokens = max(token_counts)

    # Calculate the average number of tokens
    average_tokens = sum(token_counts) / len(token_counts)

    # Find the 60th percentile of token counts
    sorted_token_counts = sorted(token_counts)
    index_60_percentile = int(0.6 * len(sorted_token_counts))
    tokens_60_percentile = sorted_token_counts[index_60_percentile]

    print(f"Maximum number of tokens: {max_tokens}")
    print(f"Average number of tokens: {average_tokens}")
    print(f"Number of tokens longer than 60% of entries: {tokens_60_percentile}")

# dataset_stats(dataset, tokenizer)

# from paper self reward paper:
# For SFT we use learning rate 5.5e−6 which linearly decays to 1.1e−6, batch size 16 and dropout 0.1

# first run - ift-2 without "gate_proj" in target_modules
# lora_dropout = 0.1
# lora_alpha=16
# lora_r=8
# learning_rate = 5.5e-6

# from https://www.datacamp.com/tutorial/mistral-7b-tutorial
# ift-3 with "gate_proj" in target_modules
lora_dropout=0.1
lora_alpha=16
lora_r=64
learning_rate=2e-4

# to try - https://www.philschmid.de/fine-tune-llms-in-2024-with-trl
# lora_dropout=0.05
# lora_alpha=128,
# lora_r=256,
# learning_rate=2e-4

batch_size = 4

def create_peft_config(model):
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        lora_dropout=lora_dropout,
        lora_alpha=lora_alpha,
        r=lora_r,        
        bias="none",
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"]
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    
    model.print_trainable_parameters()

    return model, peft_config

model, lora_config = create_peft_config(base_model)

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=batch_size,
    learning_rate=learning_rate,

    gradient_accumulation_steps=4,
    warmup_steps=30,
    logging_steps=1,
    num_train_epochs=2,
    save_steps=50,
    max_steps=350

    # max_grad_norm=0.3,
    # gradient_checkpointing=True,
    # gradient_checkpointing_kwargs={'use_reentrant':True}
)

# dataset_stats returns max. 2787, average 358, 60th percentile 371
max_seq_length = 1024

trainer = SFTTrainer(
    model=base_model,
    train_dataset=dataset,
    peft_config=lora_config,
    # dataset_text_field="prompt_response",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_args,
)

trainer.train()

output_dir = os.path.join(output_dir, "final_checkpoint")
trainer.model.save_pretrained(output_dir)