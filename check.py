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

from datasets import load_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


xquad_dataset = load_dataset('google/xquad','xquad.en')
xquad_dataset_hi =load_dataset('google/xquad','xquad.hi')


#checkpoint = "bigscience/bloomz-560m"
#checkpoint = "bigscience/bloomz-3b"
#checkpoint = "CohereForAI/aya-23-8B"
#checkpoint = "mistralai/Mistral-7B-Instruct-v0.3"
checkpoint = "meta-llama/Meta-Llama-3-8B-Instruct"
#checkpoint = 'google/gemma-7b-it'

tokenizer = AutoTokenizer.from_pretrained(checkpoint,padding_side='left')
model = AutoModelForCausalLM.from_pretrained(checkpoint,).to(device)

print(xquad_dataset['validation'][10]['question'])
print(xquad_dataset_hi['validation'][10]['question'])

context_en = xquad_dataset['validation'][34]["context"]
context_hi = xquad_dataset_hi['validation'][34]['context']
### Question:{xquad_dataset_hi['validation'][34]['question']}


Target_lang = "Hindi"

input_prompt = f"""
                    Generate an answer in Hindi based on the context below. Keep the answer short and concise.
                    ### Question:{xquad_dataset_hi['validation'][34]['question']}
                    ### Context:{context_hi}
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
outputs = model.generate(input_ids ,max_length =500,eos_token_id = terminators)

gen_tokens = [
      gt[prompt_padded_len:] for gt in outputs
    ]
gen_text = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)

#inputs = tokenizer.encode(input_ids, return_tensors="pt").to(device)
#print(tokenizer.decode(outputs[0]))


print(gen_text)
#print(tokenizer.decode(outputs[0]))









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
