import torch
import os
from .model import GPTConfig
import random
from datasets import load_dataset

MAX_LEN = 2**12 # 2**13 -> 8192

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
    def __init__(self, tokenizer, context_len, ct_extend=1):
        self.data = load_dataset("wikipedia", "20220301.en")["train"]
        self.tokenizer = tokenizer
        self.context_len = context_len
        self.ct_extend = 1
        self.train_idx = int(0.95 * len(self.data))

    def get_batch(self, split, batch_size, device):
        # generate a small batch of data of inputs x and targets y
        N = len(self.data)
        ct_len = self.context_len
        if self.ct_extend != 1:
            #ct_len *= torch.randint(
            #    low=1, high=self.ct_extend + 1, size=(1,), device=device
            #)
            ct_len = int((ct_len * self.ct_extend) * random.uniform(0.1, 1.0))

        while True:
            lo = 0 if split == "train" else self.train_idx
            hi = self.train_idx if split == "train" else N
            data = self.data[torch.randint(low=lo, high=hi, size=(1,))]["text"][0]
            if len(data) - ct_len > 0:
                break

        ix = torch.randint(len(data) - ct_len, (batch_size,))

        x = torch.tensor(
            [self.tokenizer.encode(data[i : i + ct_len]) for i in ix],
            dtype=torch.long,
            device=device,
        )
        y = torch.tensor(
            [self.tokenizer.encode(data[i + 1 : i + ct_len + 1]) for i in ix],
            dtype=torch.long,
            device=device,
        )
        return x, y


class TinyTextBook(WikiData):
    # Huggingface: https://huggingface.co/datasets/nampdn-ai/tiny-strange-textbooks
    def __init__(self, tokenizer, context_len, ct_extend=1):
        # from huggingface_hub import login
        self.data = load_dataset("nampdn-ai/tiny-strange-textbooks")["train"]
        self.tokenizer = tokenizer
        self.context_len = context_len
        self.ct_extend = ct_extend
        self.train_idx = len(self.data)

    def get_batch(self, split, batch_size, device):
        return super().get_batch("train", batch_size, device)


class OpenOrca:
    """
    This data generator generates data in-order:
        Text: <S>You are good AI</S><Q>Who are you?</Q><A>I am a good AI</A>
        Context len: 8
        
        Outputs:
            1: Inp: <S>You are good AI</S><Q>Who are you?</Q> |
               Out: <S>You are good AI</S><Q>Who are you?</Q> |
            2: Inp: <S>You are good AI</S><Q>Who are you?</Q> | <A>I
               Out: 
            3: <S>You are good AI</S><Q>Who are you?</Q> | <A>I 
            4: <S>You are good AI</S><Q>Who are you?</Q> | <A>I a
            4: <S>You are good AI</S><Q>Who are you?</Q> | <A>I am
        
    """
    # Huggingface: https://huggingface.co/datasets/Open-Orca/OpenOrca
    def __init__(self, tokenizer, context_len):
        self.data = load_dataset("Open-Orca/OpenOrca")["train"]
        self.context_len = context_len
        self.len = len(self.data["id"])
        self.tokenizer = tokenizer

    def process(self):
        """
        Tokens:
        <?>  [0] : Unknown character
        <P>  [75]: Padding
        <S>  [76]: System prompt start,
        </S> [77]: System prompt end
        <Q>  [78]: Question start
        </Q> [79]: Question end
        <A>  [80]: Answer start
        </A> [81]: Answer end
        Processes the output as follows:
        <S>SYSTEM_PROMPT</S> [DO NOT HAVE IT FOR NOW]
        <Q>QUESTION</Q>
        <A>ANSWER</A>
        """
        # Get random chat data
        idx = random.randint(0, self.len-1)
        data = self.data[idx]
        # Add system prompt and question
        text = (
            #[76]
            #+ self.tokenizer.encode(data["system_prompt"])
            #+ [77]
            #+ self.tokenizer.encode("\n")
            [78]
            + self.tokenizer.encode(data["question"])
            + [79]
            + self.tokenizer.encode("\n")
        )
        query_len = len(text)
        # Add the response
        text += [80] + self.tokenizer.encode(data["response"]) + [81]
        st = min(query_len + self.context_len, len(text))
        ed = len(text)
        
        # NOT GO BEYOND MAX_LEN
        if st >= MAX_LEN:
            return self.process()
        if st >= ed-2:
            return text, st
        
        return text, random.randint(st, min(ed-2, MAX_LEN))

    def get_batch(self, split, batch_size, device):
        x, y = [], []
        for i in range(batch_size):
            text, ed = self.process()
            x.append(text[0 : ed])
            y.append(text[1 : ed + 1])

        return (
            torch.tensor(x, dtype=torch.long, device=device),
            torch.tensor(y, dtype=torch.long, device=device),
        )


if __name__ == "__main__":
    from tokenizer import Tokenizer

    tokenizer = Tokenizer()
    data = OpenOrca(tokenizer, 128)

    for i in range(2):
        t, s = data.get_batch("train", 1, "cpu")
        print(t.shape, s.shape)
        for x, y in zip(t, s):
            x, y = x.tolist(), y.tolist()
            print("".join(tokenizer.decode(x)))
            print("==")
            print("".join(tokenizer.decode(y)))
            print("-+" * 20)

    # print(data.get_batch("train", 8, 'cpu'))
    # print(data.get_batch("test", 8, 'cpu'))
