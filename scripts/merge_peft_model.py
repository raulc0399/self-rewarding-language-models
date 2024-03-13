
# will not work as expected: https://medium.com/@bnjmn_marie/dont-merge-your-lora-adapter-into-a-4-bit-llm-65b6da287997


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import pandas as pd
from peft import PeftModelForCausalLM
import os

datasets_dir = "../datasets"

# use for simple prompting
ift_test_file_path = os.path.join(datasets_dir, "ift_test.jsonl")

# use for llm_as_a_judge_prompt.txt
eft_test_file_path = os.path.join(datasets_dir, "eft_test.jsonl")

# if using llama
access_token = ".."

device = "cuda" # the device to load the model onto

def get_bnb_config():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    return bnb_config

def merge_fined_tuned_model_and_upload():
    # the m1 model
    base_model_name = "mistralai/Mistral-7B-v0.1"
    fine_tuned_model_name = "raulc0399/mistral-7b-m1-v1"

    bnb_config = get_bnb_config()

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        quantization_config=bnb_config,
    )

    finetuned_model = PeftModelForCausalLM.from_pretrained(
        base_model,
        fine_tuned_model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        fine_tuned_model_name,
        device_map="auto",
    )

    merged_model = finetuned_model.merge_and_unload()

    # merged_model.push_to_hub("mistral-7b-m1-v1-merged")
    # tokenizer.push_to_hub("mistral-7b-m1-v1-merged")

    merged_model.save_pretrained("../merged-model/mistral-7b-m1-v1-merged")
    tokenizer.save_pretrained("../merged-model/mistral-7b-m1-v1-merged")

merge_fined_tuned_model_and_upload()