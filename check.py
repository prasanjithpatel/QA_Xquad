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


#checkpoint = "bigscience/bloomz-560m"
#checkpoint = "bigscience/bloomz-3b"
checkpoint = "CohereForAI/aya-23-8B"

tokenizer = AutoTokenizer.from_pretrained(checkpoint,padding_side='left')
model = AutoModelForCausalLM.from_pretrained(checkpoint,).to(device)



input_prompt = f"""
                    answer the question based on the context below.Answer in the same language used in the query Keep the answer short and concise.
                    ### Question:{xquad_dataset['validation'][0]['question']}
                    ### Context:{xquad_dataset['validation'][0]['context']}
                
                    """

messages = [{"role": "user", "content": str(input_prompt)}]
input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(device)
prompt_padded_len = len(input_ids[0])
print(prompt_padded_len)
outputs = model.generate(input_ids ,max_length =500)

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
