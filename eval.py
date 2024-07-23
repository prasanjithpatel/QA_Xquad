
import random 
import json 
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader, Dataset
from Dataloder import CustomDataset
from eval_metric import token_f1_score,exact_match_score


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#checkpoint = "bigscience/bloomz-560m"
#checkpoint = "bigscience/bloomz-3b"
#checkpoint = "google/gemma-7b-it"
#checkpoint = "meta-llama/Meta-Llama-3-8B"
checkpoint = "CohereForAI/aya-23-8B"
#checkpoint = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(checkpoint,padding_side='left')
model = AutoModelForCausalLM.from_pretrained(checkpoint,).to(device)


if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))



xquad_dataset = load_dataset('google/xquad','xquad.hi') #hi
val_dataset = xquad_dataset["validation"].select(range(300))
val_dataset = CustomDataset(val_dataset,tokenizer,k_shot = 0,max_length =500, model = "Aya")
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
print(len(val_dataset))



ground_truth = []
predictions = []
count = 0

with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids']
        Answers = batch["answer"]
        prompt_padded_len = len(input_ids[0])


        output = model.generate(input_ids, max_new_tokens=500, num_return_sequences=1)
        gen_tokens = [
      gt[prompt_padded_len:] for gt in output
    ]
        model_output = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)

        # Debugging print statements

        print(model_output)

        for i in range(len(model_output)):
            predictions.append(model_output[i])
            ground_truth.append(Answers['text'][0][i])

        '''
        for i, (input_text, generated_text) in enumerate(zip(input_ids, model_output)):
            decoded_input = tokenizer.decode(input_text, skip_special_tokens=True)
            print(f"Input {i}: {decoded_input}")
            print(f"Generated Output {i}: {generated_text}")
            print(generated_text.split('\n')[-1].replace("</s>", ""),"iam generated")'''

        '''

        for i in range(len(model_output)):
            ground_truth.append(Answers["text"][0][i])
            if count<=10:
                print(model_output[i].split('\n')[-1].replace("</s>", ""),)
                count+=1
            predictions.append(model_output[i].split('\n')[-1].replace("</s>", ""))'''











f1_scores = [token_f1_score(p, g) for p, g in zip(predictions, ground_truth)]

# Calculate Exact Match scores
em_scores = [exact_match_score(p, g) for p, g in zip(predictions, ground_truth)]

# Convert F1 scores to the requested format
formatted_f1_scores = ','.join(f"{score:.1f}" for score in f1_scores)
average_f1_score = sum(f1_scores) / len(f1_scores)
formatted_average_f1_score = f"{average_f1_score:.1f}"

# Convert Exact Match scores to the requested format
formatted_em_scores = ','.join(f"{score}" for score in em_scores)
average_em_score = sum(em_scores) / len(em_scores)
formatted_average_em_score = f"{average_em_score:.1f}"

# Print the results
#print("Token-level F1 Scores:", formatted_f1_scores)
print("Average Token-level F1 Score:", formatted_average_f1_score)
#print("Exact Match Scores:", formatted_em_scores)
print("Average Exact Match Score:", formatted_average_em_score)
