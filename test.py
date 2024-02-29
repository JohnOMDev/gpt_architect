#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 07:32:40 2024

@author: johnomole
"""

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
    batch_size:int = 256
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 2
    n_head: int = 2
    n_embd: int = 384
    dropout: float = 0.2
    learning_rate:float = 3e-4
    max_iters:int = 1000
    eval_iters:int = 384
    eval_interval:int = 250
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster





def get_response(prompt):
    # encoding with tiktoken
    config = GPTConfig()
    model = BigramLanguageModel(config)
    with open("model-gpt-01.pkl", 'rb') as f:
        model = pickle.load(f)
    
    m =model.to(device)
    print(enc.decode(m.generate(context, max_new_tokens=100)[0].tolist()))

if __name__ == "__main__":
    prompt =sys.argv[1]
    prompt = input("Prompt:\n")
    chars=sorted(list(set(prompt)))
    enc =tiktoken.get_encoding('gpt2')  
    
    # generate from the model
    context = torch.zeros(enc.encode_ordinary(prompt), dtype=torch.long, device=device)
    get_response(context)
    
        