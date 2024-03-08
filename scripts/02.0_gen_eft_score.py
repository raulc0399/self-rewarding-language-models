# uses the ift trained mistral model raulc0399/mistral-7b-ift-3 to generate scores for the eft dataset
# adds the following fields: generated_answer, generated_score, diff_score (between generated_score and quality_score of the original set)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import pandas as pd
import os
import re

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
    # base_model_name = "raulc0399/mistral-7b-ift-2"
    base_model_name = "raulc0399/mistral-7b-ift-3"

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
            max_new_tokens=100 # since the score is at the beginning
        )
    
        # print(f"Q: {prompt}:")
        # print("-------------------------")

        decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        answer = decoded[0]
        # print(f"A: {answer}")

        # print("\n\n")

        return answer


model, tokenizer = load_fined_tuned()
model.eval()

datasets_dir = "../datasets"
file_path = os.path.join(datasets_dir, "00_eft.jsonl")
output_file_path = os.path.join(datasets_dir, "02.0_eft_with_generated_score.jsonl")

df = pd.read_json(path_or_buf=file_path, lines=True)

file = open('llm_as_a_judge_prompt.txt', 'r')
llm_as_a_judge_prompt_template = file.read()
file.close()

pattern = r"[Ss]core: ([0-5])"
for index, row in df.iterrows():
    prompt = row['prompt_text']
    response = row['response_text']
    
    llm_as_a_judge_prompt = llm_as_a_judge_prompt_template.format(prompt=prompt,response=response)
    answer = do_sample(model, tokenizer, llm_as_a_judge_prompt)

    print("-------------------------")
    print(f"Processing index: {index} of {len(df)}")

    matches = re.findall(pattern, answer)
    generated_score = int(matches[0]) / 5 if matches else -5
    
    print("Found Score: ", generated_score)
    print(f"score from eft dataset: {row['quality_score']}")

    df.at[index, 'generated_answer'] = answer
    df.at[index, 'generated_score'] = generated_score
    df.at[index, 'diff_score'] = abs(generated_score - row['quality_score'])

    if index % 20 == 0:
        df.to_json(output_file_path, orient='records', lines=True)
        # break

df.to_json(output_file_path, orient='records', lines=True)

print("Done!")