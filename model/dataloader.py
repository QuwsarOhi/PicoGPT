import torch
import os
from .model import GPTConfig
from datasets import load_dataset


class TinyShakespere:
    def __init__(self, tokenizer):
        datapath = os.path.join(".", "dataset", "tinyshakespeare.txt")
        with open(datapath, "r", encoding="utf-8") as f:
            text = f.read()
        data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        n = int(0.9 * len(data))  # first 90% will be train, rest val
        self.train_data = data[:n]
        self.val_data = data[n:]

    def get_batch(self, split, batch_size, device):
        # generate a small batch of data of inputs x and targets y
        data = self.train_data if split == "train" else self.val_data
        ix = torch.randint(len(data) - GPTConfig.context_len, (batch_size,))
        x = torch.stack([data[i : i + GPTConfig.context_len] for i in ix])
        y = torch.stack([data[i + 1 : i + GPTConfig.context_len + 1] for i in ix])
        x, y = x.to(device), y.to(device)
        return x, y


class WikiData:
    def __init__(self, tokenizer):
        self.data = load_dataset("wikipedia", "20220301.en")["train"]
        self.tokenizer = tokenizer
        self.train_idx = int(0.95 * len(self.data))

    def get_batch(self, split, batch_size, device):
        # generate a small batch of data of inputs x and targets y
        N = len(self.data)

        while True:
            lo = 0 if split == "train" else self.train_idx
            hi = self.train_idx if split == "train" else N
            data = self.data[torch.randint(low=lo, high=hi, size=(1,))][
                "text"
            ][0]
            if len(data) - GPTConfig.context_len > 0:
                break

        ix = torch.randint(len(data) - GPTConfig.context_len, (batch_size,))

        x = torch.tensor(
            [self.tokenizer.encode(data[i : i + GPTConfig.context_len]) for i in ix],
            dtype=torch.long,
        )
        y = torch.tensor(
            [
                self.tokenizer.encode(data[i + 1 : i + GPTConfig.context_len + 1])
                for i in ix
            ],
            dtype=torch.long,
        )
        x, y = x.to(device), y.to(device)
        return x, y



class TinyTextBook(WikiData):
    # Huggingface: https://huggingface.co/datasets/nampdn-ai/tiny-strange-textbooks
    def __init__(self, tokenizer):
        #from huggingface_hub import login
        self.data = load_dataset("nampdn-ai/tiny-strange-textbooks")["train"]
        self.tokenizer = tokenizer
        self.train_idx = len(self.data)
        
    def get_batch(self, split, batch_size, device):
        return super().get_batch("train", batch_size, device)
        
        
if __name__ == "__main__":
    from tokenizer import Tokenizer
    data = TinyTextBook(Tokenizer())
    
    print(data.get_batch("train", 8, 'cpu'))
    print(data.get_batch("test", 8, 'cpu'))
        