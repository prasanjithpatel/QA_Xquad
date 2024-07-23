

import torch
from torch.utils.data import Dataset
import random


#answer the question based on the context below.Answer in the same language used in the query Keep the answer short and concise.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CustomDataset(Dataset):
    def __init__(self,data,tokenizer,k_shot,max_length,model):
        self.data = data
        self.tokenizer = tokenizer
        self.k_shot = k_shot
        self.max_length = max_length
        self.model = model 


    def __len__(self):
        return len(self.data)

    def get_prompt(self,question,context):
        if self.k_shot==0:
           input_prompt = f"""
                    answer the question based on the context below.Answer in the same language used in the query keep the answer short and concise.
                    ### Question:{question}
                    ### Context:{context}
                    ###Answer:
                    """
        return input_prompt




    def __getitem__(self,idx):
        example = self.data[idx]
        context = example['context']
        question = example['question']
        answer = example['answers']

        '''indic_content = example['indic_context']
        eng_content = example['Eng_context']
        indic_question = example['indic_question']
        eng_question = example['Eng_question']
        indic_answer = example['indic_answer']
        eng_answer = example['Eng_answer']'''

        prompt = self.get_prompt(question,context)

        if self.model == "Aya":
            messages = [{"role": "user", "content": str(prompt)}]
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







