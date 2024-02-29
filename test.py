#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 07:32:40 2024

@author: johnomole
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
import pickle
import argparse
from dataclasses import dataclass

# from train import BigramLanguageModel

parser = argparse.ArgumentParser(description='Mini GPT')

args = parser.parse_args()

device ='cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(1337)
block_size =64
batch_size =256
max_iters = 200
eval_interval = 10
learning_rate = lr=3e-4
eval_iters = 384
device ='cuda' if torch.cuda.is_available() else 'cpu'
n_embd = 384
n_head = 2
n_layer = 2
dropout =0.2




class Head(nn.Module):
    def __init__(self, head_size, config):
        super().__init__()
        self.key = nn.Linear(config.n_embd, head_size, bias=False)
        self.query = nn.Linear(config.n_embd, head_size, bias=False)
        self.value = nn.Linear(config.n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))
        
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x) # B,T, C
        q = self.query(x) # B, T, C
        # COMPUTE ATTENTION scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B,T,C)  @ (B, C, T) --> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T]==0, float('-inf')) #(B, T, T)
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""
    def __init__(self, num_heads, head_size, config):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, config) for _ in range(num_heads)])
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
    def forward(self, x):
        out =torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """
        a simple linear layer followed by a non-linearity
    """
    def __init__(self, n_embd, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(config.dropout)
            )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transfomer block: communication followed by computation """
    def __init__(self, config, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, config)
        self.ffwd = FeedForward(n_embd, config)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
        
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        
        return x


@dataclass
class GPTConfig:
    block_size: int = 64
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 2
    n_head: int = 2
    n_embd: int = 384
    dropout: float = 0.2
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster


class BigramLanguageModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.config = config
        self.token_embedding_table =nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(
                *[Block(config, config.n_embd, n_head=config.n_head) for _ in range(config.n_layer)]
            )
        self.ln_f = nn.LayerNorm(config.n_embd) # final layer norm
        self.lm_head =nn.Linear(config.n_embd, config.vocab_size)
        
        
    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        tok_emb =self.token_embedding_table(idx) #(B T C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) #(B, T, C)
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B,T,vocab_size)
        if targets is None:
            loss=None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        #idx is (B,T) array of indeces in the current context
        
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, - self.config.block_size:]
            # get predictions
            logits, loss = self(idx_cond)
            logits = logits[:,-1, :]
            probs =F.softmax(logits, dim=-1)
            idx_next =torch.multinomial(probs, num_samples=1)
            idx =torch.cat((idx, idx_next), dim=1)
        return idx
    
    

config = GPTConfig()


model = BigramLanguageModel(config)



with open("model-gpt-01.pkl", 'rb') as f:
    model = pickle.load(f)

# def get_response(prompt):
    # encoding with tiktoken
m =model.to(device)


if __name__ == "__main__":
    prompt = input("Prompt:\n")
    chars=sorted(list(set(prompt)))
    enc =tiktoken.get_encoding('gpt2')  
    
    # generate from the model
    context = torch.zeros(enc.encode_ordinary(prompt), dtype=torch.long, device=device)
    print(enc.decode(model.generate(context, max_new_tokens=100)[0].tolist()))
        