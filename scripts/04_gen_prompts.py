import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import pandas as pd

import os

access_token = ".."

device = "cuda" # the device to load the model onto

num_prompts_to_generate=500

model_id_mistral_instruct = "mistralai/Mistral-7B-Instruct-v0.2"
model_id_mistral_llama_chat = "meta-llama/Llama-2-13b-chat-hf"

# Define constants for file paths
datasets_dir = "../datasets"

# the ift dataset used in 01_and_03_sft.py
ift_file_path = os.path.join(datasets_dir, "ift.jsonl")

# location of the generated prompts
generated_prompts_file_path = os.path.join(datasets_dir, "generated_prompts.jsonl")

def read_jsonl_file(file_path):
    """Read a JSONL file into a pandas DataFrame."""
    return pd.read_json(file_path, lines=True)

def save_to_jsonl(df, file_path):
    """Save a DataFrame to a JSONL file."""
    df.to_json(file_path, orient='records', lines=True)

def get_bnb_config():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    return bnb_config

def load_model(model_id):
    
    bnb_config = get_bnb_config()

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        token=access_token
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=access_token
    )
    model.eval()

    return model, tokenizer

def generate_task_array_for_prompt(example_tasks):
    prompt_array = []
    for index, item in enumerate(example_tasks):
        user_content = f"Task {index + 1}:"
        assistant_content = item
        
        if index == 0:
            user_content = "Come up with a series of tasks, only the task/question, no further text/explanation, no additional information: " + user_content
        
        prompt_array.append({"role": "user", "content": user_content})
        prompt_array.append({"role": "assistant", "content": assistant_content})

        if index == len(example_tasks) - 1:
            user_content = f"Task {index + 2}:"
            prompt_array.append({"role": "user", "content": user_content})
    
    return prompt_array

def do_sample(model, tokenizer, task_prompts):
    with torch.no_grad():
        prompt_for_new_task = generate_task_array_for_prompt(task_prompts)
        # print(f"Prompt for new task: {prompt_for_new_task}")
        
        print("-----------------------------------------------------------------------")
        prompt_for_model = tokenizer.apply_chat_template(prompt_for_new_task, tokenize=False)
        # print(f"Prompt for model: {prompt_for_model}")

        model_inputs = tokenizer(prompt_for_model, return_tensors="pt").to("cuda")

        generated_ids = model.generate(
            **model_inputs,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1,
            # top_k=50,
            top_p=0.9,
            temperature=0.6,
            max_new_tokens=1000
        )
    
        decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        answer = decoded[0]
        # print(f"A: {answer}")

        # print("\n\n")

        return answer

def get_random_prompts(df, num_selections=8):
    all_selected_prompts = df.sample(n=num_selections)['prompt'].tolist()

    return all_selected_prompts

def extract_prompt_only(answer, task_nr):
    pattern = f"[INST] Task {task_nr}: [/INST]"
    parts = answer.split(pattern)
    if len(parts) > 1:
        return parts[1]
    else:
        return ""

model, tokenizer = load_model(model_id_mistral_instruct)

ift_df = read_jsonl_file(ift_file_path)
new_prompts = []
for i in range(num_prompts_to_generate):
    print(f"Generating prompt {i + 1} of {num_prompts_to_generate}...")

    task_prompts = get_random_prompts(ift_df)

    answer = do_sample(model, tokenizer, task_prompts)
    prompt_only = extract_prompt_only(answer, len(task_prompts) + 1)
    
    print(f"Answer: {answer}")
    print(f"\n\nPrompt only: {prompt_only}")

    new_prompts.append({"prompt": prompt_only})

new_prompts_df = pd.DataFrame(new_prompts)
save_to_jsonl(new_prompts_df, generated_prompts_file_path)