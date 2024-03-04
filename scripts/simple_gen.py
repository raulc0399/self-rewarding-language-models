import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import pandas as pd
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

def load_mistral():
    base_model_name = "mistralai/Mistral-7B-v0.1"
    # base_model_name = "meta-llama/Llama-2-13b-chat-hf"

    bnb_config = get_bnb_config()

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        token=access_token
    )
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        token=access_token
    )

    return model, tokenizer

def load_fined_tuned():
    # ift fine-tuned model
    # base_model_name = "raulc0399/mistral-7b-ift-3"

    # the m1 model
    base_model_name = "raulc0399/mistral-7b-m1-v1"

    bnb_config = get_bnb_config()

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        quantization_config=bnb_config,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        device_map="auto",
    )

    return model, tokenizer

def do_sample(model, tokenizer, prompt):
    with torch.no_grad():
        prompt_sample = [
            {"role": "user", "content": prompt},
            # {"role": "assistant", "content": ""},
        ]
        
        print("-----------------------------------------------------------------------")
        prompt_for_model = tokenizer.apply_chat_template(prompt_sample, tokenize=False)
        # print(f"Prompt for model: {prompt_for_model}")

        model_inputs = tokenizer(prompt_for_model, return_tensors="pt").to("cuda")

        generated_ids = model.generate(
            **model_inputs,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1,
            top_p=0.9,
            temperature=0.7,
            max_new_tokens=500
        )
    
        print(f"Q: {prompt}:")
        print("-------------------------")

        decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        print(f"A: {decoded[0]}")

        print("\n\n")


# model, tokenizer = load_mistral()
model, tokenizer = load_fined_tuned()

model.eval()

df_ift_test = pd.read_json(path_or_buf=ift_test_file_path, lines=True)
for index, row in df_ift_test.iterrows():
    prompt = row['prompt']
    completion = row['completion']
    
    do_sample(model, tokenizer, prompt)

    print(f"Completion from dataset: {completion}")
    print("\n\n")

file = open('llm_as_a_judge_prompt.txt', 'r')
llm_as_a_judge_prompt_template = file.read()
file.close()

df_eft_test = pd.read_json(path_or_buf=eft_test_file_path, lines=True)
for index, row in df_eft_test.iterrows():
    prompt = row['prompt_text']
    response = row['response_text']
    quality_score = row['quality_score']

    llm_as_a_judge_prompt = llm_as_a_judge_prompt_template.format(prompt=prompt,response=response)
    do_sample(model, tokenizer, llm_as_a_judge_prompt)

    print(f"Quality score from dataset: {quality_score}")
    print("\n\n")
