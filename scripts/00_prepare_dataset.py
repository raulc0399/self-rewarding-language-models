import pandas as pd
import os

datasets_dir = "../datasets"

def save_as_parquet(df):
    df['prompt_response'] = df['prompt_text'] + ' ' + df['response_text']
    df = df[['prompt_response']]

    # keep 10 for eval
    eval_df = df.sample(n=10, random_state=42)
    df = df.drop(eval_df.index)

    # print(df)
    # print(eval_df)

    df.to_parquet(os.path.join(datasets_dir, 'ift.parquet'), index=False)
    eval_df.to_json(os.path.join(datasets_dir, 'ift_test.jsonl'), index=False, orient="records", lines=True)

# https://huggingface.co/docs/trl/v0.7.11/sft_trainer#dataset-format-support
def save_as_sfttrainer_format(df):
    df.rename(columns = {'prompt_text': 'prompt', 'response_text': 'completion'}, inplace = True) 
    df = df[['prompt', 'completion']]
    
    # keep 10 for eval
    eval_df = df.sample(n=10, random_state=42)
    df = df.drop(eval_df.index)

    # print(df)
    # print(eval_df)

    df.to_json(os.path.join(datasets_dir, 'ift.jsonl'), index=False, orient="records", lines=True)
    eval_df.to_json(os.path.join(datasets_dir, 'ift_test.jsonl'), index=False, orient="records", lines=True)

def split_etf(df):
    # keep 10 for eval
    eval_df = df.sample(n=10, random_state=42)
    df = df.drop(eval_df.index)

    # print(df)
    # print(eval_df)

    df.to_json(os.path.join(datasets_dir, 'eft.jsonl'), index=False, orient="records", lines=True)
    eval_df.to_json(os.path.join(datasets_dir, 'eft_test.jsonl'), index=False, orient="records", lines=True)

file_path = os.path.join(datasets_dir, "oasst2_instruction_fine_tuning.jsonl")
df = pd.read_json(path_or_buf=file_path, lines=True)

save_as_sfttrainer_format(df)

file_path = os.path.join(datasets_dir, "oasst2_evaluation_fine_tuning.jsonl")
df = pd.read_json(path_or_buf=file_path, lines=True)

split_etf(df)