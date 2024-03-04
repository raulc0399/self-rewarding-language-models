**00_prepare_dataset.py**\
takes the ift dataset and saves it in the format for hf sft trainer, keeps 10 for testing
also splits the eft dataset, keeps 10 for testing

**01_and_03_sft.py**
runs sft training on mistral Mistral-7B-v0.1
in step 1 using only ift.json
in step 3 using the ift + the generated eft set (the one containing the llm as judge promt + the completion for score)

**02.0_gen_eft_score.py**
using the ift trained model (based on mistral) and llm_as_a_judge_prompt it asks to generate the scrore for the prompt and completion from the eft dataset

**02.1_select_eft_and_merge_with_ift.py**
selects the entries where score generated between the model fine-tuned in step 1 and the score from the original eft dataset have a diff of max. 0.25
saves them as prompt and completin format for sft trainer. the prompt is the llm_as_a_judge_prompt prompt and the completion is the score from the eft dataset converted to 1 to 5 range
the initial ift dataset is merged with this one

**01_and_03_sft.py**
sft training is run with the new ift + prepared eft dataset
this will generate the m1 model

**04_gen_prompts.py**
using Mistral-7B-Instruct-v0.2 and the initial ift dataset, as well as a 8-shot prompt (based on the entries from the ift dataset) it generates new prompts to be used in the next 2 loops

**05.0_gen_responses.py**
this and the next steps are the ones run in the 2 loop, ones with m1 and ones with m2
using the m1 model it generates 4 responsess for each of the previously generated prompts

**05.1_gen_scores.py**
using the m1 model it generates the scores for the propmts and generated responses

**05.2_gen_preferences_pairs.py**
generates the set to be used in dpo training, by keeping for each prompt only the responses with the min. and max. score

**06_dpo.py**
will use the dataset generated in the previous step to run dpo training on the m1 model, this will result in m2
running the steps 5 and 6 again will generate the m3 model

**llm_as_a_judge_prompt_orig.txt**
the prompt 

**llm_as_a_judge_prompt.txt**
was changed to generate responses containint the score at first, in the format that can be post-processed

simple_gen.py