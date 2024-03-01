#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 07:32:40 2024

@author: johnomole
"""
import io
import torch
import tiktoken
import pickle
from model import BigramLanguageModel
from dataclasses import dataclass
import sys
device ='cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(1337)



@dataclass
class GPTConfig:
    block_size: int = 64
    batch_size:int = 100
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.2
    learning_rate:float = 3e-2
    max_iters:int = 5000
    eval_iters:int = 384
    eval_interval:int = 1000



class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)



def get_response(context):
    # encoding with tiktoken
    config = GPTConfig()
    model = BigramLanguageModel(config)
    with open("model-gpt-01.pkl", 'rb') as f:
        model = CPU_Unpickler(f).load()
    
    m =model.to(device)
    
    print(enc.decode(m.generate(context.unsqueeze(0), max_new_tokens=100)[0].tolist()))

if __name__ == "__main__":
    prompt =sys.argv[1]
    # prompt = "I enjoyed the crispy chicken at KFC because"
    enc =tiktoken.get_encoding('gpt2')
    # generate from the model
    context = torch.tensor(enc.encode_ordinary(prompt), dtype=torch.long, device=device)
    get_response(context)
    
        