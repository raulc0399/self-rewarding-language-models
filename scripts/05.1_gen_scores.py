import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import pandas as pd
import re
import os

datasets_dir = "../datasets"

# generated prompts with 4 responses and scores
generated_prompts_responses_file_path = os.path.join(datasets_dir, "05.0_generated_prompts_responses.jsonl")

# generated prompts with 4 responses and scores
generated_prompts_responses_with_scores_file_path = os.path.join(datasets_dir, "05.1_generated_prompts_responses_with_scores.jsonl")

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

df = pd.read_json(path_or_buf=generated_prompts_responses_file_path, lines=True)

file = open('llm_as_a_judge_prompt.txt', 'r')
llm_as_a_judge_prompt_template = file.read()
file.close()

# !! since the eft_selected_prepared_dataset contained the score as float => use it as float here
pattern = r"[Ss]core: ([0-5])"
for index, row in df.iterrows():
    prompt = row['prompt']
    
    for i_c, completion_entry in enumerate(row['completions']):
        
        print("-------------------------")
        print(f"Processing index: {index} completion {i_c}")

        response = completion_entry["completion"]

        llm_as_a_judge_prompt = llm_as_a_judge_prompt_template.format(prompt=prompt,response=response)
        answer = do_sample(model, tokenizer, llm_as_a_judge_prompt)

        matches = re.findall(pattern, answer)
        generated_score = int(matches[0]) if matches else -1
    
        print(f"Answer {answer}")
        print("Found Score: ", generated_score)

        completion_entry["score"] = generated_score

    if index % 20 == 0:
        df.to_json(generated_prompts_responses_with_scores_file_path, orient='records', lines=True)
        # break

df.to_json(generated_prompts_responses_with_scores_file_path, orient='records', lines=True)

print("Done!")
