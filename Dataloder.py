

import torch
from torch.utils.data import Dataset
import random


#answer the question based on the context below.Answer in the same language used in the query Keep the answer short and concise.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CustomDataset(Dataset):
    def __init__(self,data,ind_data,tokenizer,k_shot,max_length,model):
        self.data = data
        self.ind_data = ind_data
        self.tokenizer = tokenizer
        self.k_shot = k_shot
        self.max_length = max_length
        self.model = model 


    def __len__(self):
        return len(self.data)

    def get_prompt(self,question,context):
        if self.k_shot==0:
           input_prompt = f"""
                    Generate an answer in English based on the context below.keep the answer short and concise.
                    ### Question:{question}
                    ### Context:{context}
                    ### Answer:
                    """
        return input_prompt




    def __getitem__(self,idx):
        example = self.data[idx]
        ind_example = self.ind_data[idx]

        #English 
        context = example['context']
        question = example['question']
        answer = example['answers']

        #indic 
        indic_context = ind_example['context']
        indic_question = ind_example['question']
        indic_answer = ind_example['answers']

        prompt = self.get_prompt(question,context)

        if self.model != None:

            messages = [{"role": "system", "content": str(prompt)}]
            input_ids = self.tokenizer.apply_chat_template(messages, tokenize=True, 
                                                           add_generation_prompt=True, 
                                                           padding = "max_length",
                                                           truncation = True,
                                                           max_length = 512,
                                                           return_tensors="pt").squeeze(0).to(device)
            return {
                'input_ids' : input_ids,
                'answer' : answer
            }
        

        encoded_prompt = self.tokenizer(
            prompt,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=512,
            
        )
        input_ids = encoded_prompt['input_ids'].squeeze(0).to(device)


        
        return {
              'input_ids': input_ids,
              'answer': answer
      
          }







