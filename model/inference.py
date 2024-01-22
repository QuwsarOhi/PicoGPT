import torch
import torch.nn as nn
from dataclasses import dataclass
import os 
import io
import pickle
import math
import matplotlib.pyplot as plt

from model import GPT
from tokenizer import Tokenizer
from train import GPTConfig


model = GPT(GPTConfig)

tokenizer = Tokenizer()

while True:
    print("Input: ")
    x = input()
    x = tokenizer.encode(x)
    
    