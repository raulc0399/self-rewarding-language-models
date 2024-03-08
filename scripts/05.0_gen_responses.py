# uses the m1 model to generate 4 completions for each prompt

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import pandas as pd
import os

datasets_dir = "../datasets"

# use for simple prompting
prompts_file_path = os.path.join(datasets_dir, "04_generated_prompts.jsonl")

# generated prompts with 4 responses and scores
generated_prompts_responses_file_path = os.path.join(datasets_dir, "05.0_generated_prompts_responses.jsonl")

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
            temperature=0.7,
            top_p=0.9,
            max_new_tokens=200
        )
    
        print(f"Q: {prompt}:")
        print("-------------------------")

        answer = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        answer = answer[0]
        print(f"A: {answer}")

        print("\n\n")

        return answer

def extract_completion_only(answer):
    pattern = f"[/INST]"
    parts = answer.split(pattern)
    if len(parts) > 1:
        return parts[-1]
    else:
        return ""

model, tokenizer = load_fined_tuned()
model.eval()

df_prompts = pd.read_json(path_or_buf=prompts_file_path, lines=True)
# df_prompts = df_prompts.sample(100).reset_index(drop=True)

df_prompts['completions'] = [[] for _ in range(df_prompts.shape[0])]
for index, row in df_prompts.iterrows():
    print(f"Processing prompt {index + 1} of {len(df_prompts)}")
    
    prompt = row['prompt']
        
    # sample 4 times as mentioned in the paper
    for completion_sample in range(4):
        print(f"Processing prompt {index + 1}, completion {completion_sample + 1}")

        answer = do_sample(model, tokenizer, prompt)
        completion = extract_completion_only(answer)

        # -1 as not evaluated yet
        df_prompts.at[index, 'completions'].append({"completion": completion, "score": -1})
        
        # print("\n\n")
        # print(f"Extracted completion: {completion}")

    if index % 10 == 0:
        df_prompts.to_json(generated_prompts_responses_file_path, orient='records', lines=True)

df_prompts.to_json(generated_prompts_responses_file_path, orient='records', lines=True)
