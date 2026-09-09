import numpy as np
import pandas as pd

from torch import nn
import torch.nn.functional as F
import string
import random
import torch
from torchviz import make_dot
import time
from torchinfo import summary

import json
import csv
import string

dims = 512
n_layers = 12
head_dim = 64
hidden_dim = 0
n_heads = 8
ntokens = 3200
vocab_size = 11157
dropout_rate = 0.0

class Head(nn.Module):
    def __init__(self):
        super().__init__()

        self.key = nn.Linear(dims, head_dim, bias=False)
        self.query = nn.Linear(dims, head_dim, bias=False)
        self.value = nn.Linear(dims, head_dim, bias=False)

        self.register_buffer("tril", torch.tril(torch.ones(ntokens, ntokens)))

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        B, T, C = x.shape
        
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
    
        wei = q @ k.transpose(-2, -1) * head_dim **-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        out = wei @ v

        return out


class MultiHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.heads = nn.ModuleList([Head() for i in range(n_heads)])
        self.lin = nn.Linear(dims, dims)
        self.dropout = nn.Dropout(dropout_rate)


    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.lin(out))

        return out

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dims, 4*dims),
            nn.ReLU(),
            nn.Linear(4*dims, dims),
            nn.Dropout(dropout_rate)
        )
    def forward(self, x):
        out = self.net(x)
        return out

class Block(nn.Module):
    def __init__(self):
        super().__init__()

        self.mha = MultiHead()
        self.ffw = FeedForward()
        self.ln1 = nn.LayerNorm(dims)
        self.ln2 = nn.LayerNorm(dims)

    def forward(self,x):
        x = x + self.mha(self.ln1(x))
        x = x + self.ffw(self.ln2(x))

        return x
        

class FinalModel(nn.Module):
  def __init__(self):
    super().__init__()

    self.token_embedding = nn.Embedding(vocab_size, dims)
    self.positional_embedding = nn.Embedding(ntokens, dims)

    self.blocks = nn.Sequential(*[Block() for _ in range(n_layers)])

    self.ln = nn.LayerNorm(dims)
    self.lin = nn.Linear(dims, vocab_size)



  def forward(self, x):
    B, T = x.shape
    token_emd = self.token_embedding(x.long())
    posi = self.positional_embedding(torch.arange(0, T, device=device))

    x = token_emd + posi

    x = self.blocks(x)
    x = self.ln(x)
    out = self.lin(x)

    return out
