from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SpecialTokens:
    pad_token: str
    end_token: str
    male_token: str
    female_token: str


@dataclass
class Tokenizer:
    idx2token: dict[int,str]
    token2idx: dict[str,int]
    special_tokens: SpecialTokens

    def encode(self, text: str, gender: Literal["Male"] | Literal["Female"], max_len=24) -> list[int]:

        if gender == "Male":
            gender_token = self.special_tokens.male_token
        elif gender == "Female":
            gender_token = self.special_tokens.female_token
        else:
            raise RuntimeError("Invalid gender!!!")
        
        name = [gender_token] + [*text[:max_len]]
        name.append(self.special_tokens.end_token)
        
        while len(name) < max_len:
            name.append(self.special_tokens.end_token)
            
        return [self.token2idx[c] for c in name]


    def decode(self, tokens: list[int]) -> str:
        return "".join([self.idx2token[token] for token in tokens])


@dataclass
class RNNConfig:
    vocab_size: int
    embed_dim: int
    hidden_dim: int
    device: Literal["cuda:0"] | Literal["cpu"]

class RNNModel(nn.Module):
    
    def __init__(self, config: RNNConfig):
    
        super().__init__()
        self.config = config

        self.W_hh = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.W_xh = nn.Linear(config.embed_dim, config.hidden_dim)
        self.W_hy = nn.Linear(config.hidden_dim, config.vocab_size)
        
        self.h = nn.Parameter(torch.randn(config.hidden_dim))
        self.embeddings = nn.Embedding(config.vocab_size, config.embed_dim)
        
    def forward(self, x):


        x = self.embeddings(x)
        batch_size, seq_len, _ = x.shape


        output = torch.zeros(batch_size, seq_len - 1, self.config.vocab_size).to(self.config.device)
        hiddens = torch.zeros(batch_size, self.config.hidden_dim).to(self.config.device)
        
        for i in range(batch_size):
            hiddens[i] = self.h
        
        for i in range(seq_len - 1):
            
            hiddens = F.tanh(self.W_hh(hiddens) + self.W_xh(x[:,i] + x[:,0]))
            y = self.W_hy(hiddens)
            output[:,i] = y
        
        return output

