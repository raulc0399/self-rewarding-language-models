# goes over each of the 4 completions and keeps only the min and max score
# these are stores in the format [prompt, chosen, rejected] expected by the dpo trainer

import pandas as pd
import os

datasets_dir = "../datasets"

# generated prompts with 4 responses and scores
generated_prompts_responses_with_scores_file_path = os.path.join(datasets_dir, "05.1_generated_prompts_responses_with_scores.jsonl")

preferences_pairs_file_path = os.path.join(datasets_dir, "05.2_preferences_pairs.jsonl")

df = pd.read_json(path_or_buf=generated_prompts_responses_with_scores_file_path, lines=True)

indices_to_delete = []
count_rows_to_delete_no_completions = 0
count_rows_to_delete_same_chosen_rejected = 0

for index, row in df.iterrows():
    print(f"Processing row {index}")

    # Sort the 'completions' based on 'score'
    sorted_completions = sorted(row['completions'], key=lambda x: x['score'])
    
    del_row = False

    # Check if there are at least two different completions to avoid duplicating the same entry
    if len(sorted_completions) > 1:
        print(f"Scores: {sorted_completions[0]['score']} - {sorted_completions[-1]['score']}")

        if sorted_completions[0]['score'] == sorted_completions[-1]['score']:
            print(f"Skipping row {index} because the chosen and rejected completions are the same")
            del_row = True
            count_rows_to_delete_same_chosen_rejected += 1
        else:
            # Keep only the completions with the minimum and maximum scores
            # if there is a -1 score, take the completion as rejected
            min_max_completions = {"rejected": sorted_completions[0]["completion"], "chosen": sorted_completions[-1]["completion"]}
    else:
        print(f"Skipping row {index} because there are not enough completions")
        del_row = True
        count_rows_to_delete_no_completions += 1

    if del_row:
        indices_to_delete.append(index)
        continue
    
    # Update the 'completions' in the DataFrame
    df.at[index, 'chosen'] = min_max_completions['chosen']
    df.at[index, 'rejected'] = min_max_completions['rejected'] if 'rejected' in min_max_completions else ""

# Drop the rows that were marked for deletion
df = df.drop(indices_to_delete)

df = df.drop(columns=['completions'])

print(f"Deleted {count_rows_to_delete_no_completions} rows with no completions")
print(f"Deleted {count_rows_to_delete_same_chosen_rejected} rows with same chosen and rejected completions")
print(f"Deleted total of rows {len(indices_to_delete)}, remaining total rows: {len(df)}")

df.to_json(preferences_pairs_file_path, orient='records', lines=True)
