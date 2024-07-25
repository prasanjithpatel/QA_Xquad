import torch
import re
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Subset, RandomSampler
from tqdm import tqdm
import random
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score

from collections import Counter
from sklearn.metrics import f1_score as sklearn_f1_score
from transformers import LlamaTokenizer, LlamaForCausalLM


from datasets import load_dataset
from eval_metric import token_f1_score,exact_match_score


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


xquad_dataset = load_dataset('google/xquad','xquad.en')
xquad_dataset_hi =load_dataset('google/xquad','xquad.hi')


#checkpoint = "bigscience/bloomz-560m"
#checkpoint = "bigscience/bloomz-3b"
#checkpoint = "CohereForAI/aya-23-8B"
#checkpoint = "mistralai/Mistral-7B-Instruct-v0.3"
checkpoint = "meta-llama/Meta-Llama-3-8B-Instruct"
#checkpoint = 'google/gemma-7b-it'




tokenizer = LlamaTokenizer.from_pretrained('sarvamai/OpenHathi-7B-Hi-v0.1-Base')
model = LlamaForCausalLM.from_pretrained('sarvamai/OpenHathi-7B-Hi-v0.1-Base', torch_dtype=torch.bfloat16).to(device)

#tokenizer = AutoTokenizer.from_pretrained(checkpoint,padding_side='left')
#model = AutoModelForCausalLM.from_pretrained(checkpoint,).to(device)


def get_out(Question,context):

    input_prompt = f"""
                        Generate an answer in English based on the context below.Give exact answer without extra information.
                        ### Question:{Question}
                        ### Context:{context}
                        ### Answer:
                    
                        """

    messages = [
        {"role": "user", "content": str(input_prompt)},
    ]

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]


#messages = [{"role": "user", "content": str(input_prompt)}]
    input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(device)


#input_ids = tokenizer.encode(input_prompt, return_tensors="pt").to(device)






    prompt_padded_len = len(input_ids[0])
    print(prompt_padded_len)
    outputs = model.generate(input_ids ,max_length =1000,eos_token_id = terminators,do_sample=True)

    gen_tokens = [
        gt[prompt_padded_len:] for gt in outputs
        ]
    gen_text = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)

#inputs = tokenizer.encode(input_ids, return_tensors="pt").to(device)
#print(tokenizer.decode(outputs[0]))


    return(gen_text)
#print(tokenizer.decode(outputs[0]))



#xquad_dataset_hi['validation'][i]["context"]+ " "

predictions = []
ground_truth = []
for i in range(300):
    context = xquad_dataset_hi['validation'][i]['context']+" " +xquad_dataset['validation'][i]['context']
    pred = (get_out(context,xquad_dataset_hi['validation'][i]["question"]))
    gt = xquad_dataset_hi['validation'][i]["answers"]['text']
    print(pred)
    print(gt)
    predictions.append(pred)
    ground_truth.append(gt)




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


'''
messages = [{"role": "user", "content": str(input_prompt)}]
input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
print(tokenizer.decode(input_ids[0]))'''









'''
outputs = model.generate(
    input_ids,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)
response = outputs[0][input_ids.shape[-1]:]
print(tokenizer.decode(response, skip_special_tokens=True))'''
