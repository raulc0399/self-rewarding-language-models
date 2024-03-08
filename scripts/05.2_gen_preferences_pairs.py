import pandas as pd
import os

datasets_dir = "../datasets"

# generated prompts with 4 responses and scores
generated_prompts_responses_with_scores_file_path = os.path.join(datasets_dir, "05.1_generated_prompts_responses_with_scores.jsonl")

preferences_pairs_file_path = os.path.join(datasets_dir, "05.2_preferences_pairs.jsonl")

df = pd.read_json(path_or_buf=generated_prompts_responses_with_scores_file_path, lines=True)

for index, row in df.iterrows():
    # Sort the 'completions' based on 'score'
    sorted_completions = sorted(row['completions'], key=lambda x: x['score'])
    
    # Check if there are at least two different completions to avoid duplicating the same entry
    if len(sorted_completions) > 1:
        # Keep only the completions with the minimum and maximum scores
        min_max_completions = {"rejected": sorted_completions[0]["completion"], "chosen": sorted_completions[-1]["completion"]}
    else:
        # In case there's only one completion, keep it as it is
        min_max_completions = {"chosen": sorted_completions[-1]["completion"]}
    
    # Update the 'completions' in the DataFrame
    df.at[index, 'chosen'] = min_max_completions['chosen']
    df.at[index, 'rejected'] = min_max_completions['rejected'] if 'rejected' in min_max_completions else ""

df = df.drop(columns=['completions'])

df.to_json(preferences_pairs_file_path, orient='records', lines=True)
