# %%
import torch
import torch.nn as nn
from dataclasses import dataclass
import os 
import io
import pickle
import copy
import math
import matplotlib.pyplot as plt

if hasattr(__builtins__,'__IPYTHON__'):
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

from model import GPT, Tokenizer

# %%
@dataclass
class GPTConfig:
    context_len: int = 256
    vocab_size: int = 128 
    n_layer: int = 8
    n_head: int = 2
    n_embd: int = 64
    dropout: float = 0.05
    bias: bool = False 
    

# %%
@dataclass
class TrainConfig:
    batch_size: int = 8
    dtype: str = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    device: str = 'gpu' if torch.cuda.is_available() else 'cpu'
    warmup_iters = 2000
    learning_rate = 6e-4
    lr_decay_iters = 600000
    min_lr = 6e-5    
    weight_decay = 1e-1
    beta1 = 0.9
    beta2 = 0.95

# %%
tokenizer = Tokenizer()
# print(tokenizer.vocab_size)
# print(tokenizer.encode("This is a sentence"))
# print(''.join(tokenizer.decode(tokenizer.encode("This is a sentence"))))
# print(''.join(tokenizer.decode([0] + tokenizer.encode("This is a sentence"))))

# %%
# TINYSHAKESPERE
# datapath = os.path.join(
#     '..', 
#     'dataset',
#     'tinyshakespeare.txt'
# )

# with open(datapath, 'r', encoding='utf-8') as f:
#     text = f.read()

# data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
# n = int(0.9*len(data)) # first 90% will be train, rest val
# train_data = data[:n]
# val_data = data[n:]

# def get_batch(split):
#     # generate a small batch of data of inputs x and targets y
#     data = train_data if split == 'train' else val_data
#     ix = torch.randint(len(data) - GPTConfig.context_len, (TrainConfig.batch_size,))
#     x = torch.stack([data[i:i+GPTConfig.context_len] for i in ix])
#     y = torch.stack([data[i+1:i+GPTConfig.context_len+1] for i in ix])
#     x, y = x.to(TrainConfig.device), y.to(TrainConfig.device)
#     return x, y

# %%
from datasets import load_dataset

wiki_data = load_dataset("wikipedia", "20220301.en")

# %%
def get_batch():
    # generate a small batch of data of inputs x and targets y
    N = len(wiki_data['train'])
    
    while True:
        data = wiki_data['train'][torch.randint(N, (1,))]['text'][0]
        if len(data) - GPTConfig.context_len > 0:
            break
        
    ix = torch.randint(len(data) - GPTConfig.context_len, (TrainConfig.batch_size,))
    x = torch.tensor([tokenizer.encode(data[i:i+GPTConfig.context_len]) for i in ix], 
                     dtype=torch.long)
    y = torch.tensor([tokenizer.encode(data[i+1:i+GPTConfig.context_len+1]) for i in ix], 
                     dtype=torch.long)
    x, y = x.to(TrainConfig.device), y.to(TrainConfig.device)
    return x, y

#print(get_batch())

# %%
model = GPT(GPTConfig)

# %%
# x, y = get_batch('train')
# print(x.shape, y.shape)
# out, loss = model(x, y)
# print(out.shape, loss)

# %%
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < TrainConfig.warmup_iters:
        return TrainConfig.learning_rate * it / TrainConfig.warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > TrainConfig.lr_decay_iters:
        return TrainConfig.min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - TrainConfig.warmup_iters) / (TrainConfig.lr_decay_iters - TrainConfig.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return TrainConfig.min_lr + coeff * (TrainConfig.learning_rate - TrainConfig.min_lr)

# x = list(range(5000000))
# y = [get_lr(xx) for xx in x]
# plt.plot(x, y)

# %%
optimizer = model.configure_optimizers(TrainConfig.weight_decay, 
                                       TrainConfig.learning_rate, 
                                       (TrainConfig.beta1, TrainConfig.beta2), 
                                       TrainConfig.device)

# %%
# https://github.com/pytorch/pytorch/issues/16797#issuecomment-633423219
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


def train_fn(model: nn.Module,
             epoch: int,
             optimizer: torch.optim.Optimizer,
             savepath: str = None,
             device='cpu'):

    best_weight = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    iter_num = 0
    train_phases = ['train', 'val']
    losses = {phase: [] for phase in train_phases}

    if savepath and os.path.exists(savepath):
        # Try loading the model and weight
        try:
            with open(savepath, 'rb') as filehandler:
                prev_train = CPU_Unpickler(filehandler).load()
                best_weight = prev_train['best_weight']
                model.load_state_dict(best_weight, strict=False)
                losses = prev_train['losses']
                best_loss = prev_train['best_loss']
                optimizer = prev_train['optimizer'] 
                iter_num = prev_train['iter_num']
                print(f"Loaded model with loss: {best_loss:0.4f}")
        except:
            print(f"Could not load from path: {savepath}")


    for e in range(epoch):
        for phase in train_phases:
            is_training = (phase == 'train')
            model.train() if is_training else model.eval()
            loss, dats = 0., 0.,
            tqdm_prog = tqdm(range(500))

            for _ in tqdm_prog:
                x, y = get_batch()
                #x, y = x.to(device), y.to(device)

                with torch.set_grad_enabled(phase == 'train'):
                    _, batch_loss = model(x, y)
                    
                    if is_training:
                        batch_loss.backward()
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        # LR scheduler
                        lr = get_lr(iter_num)
                        iter_num += 1
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr

                # Stats
                dats += x.size(0)
                loss += batch_loss.item() * x.size(0)
                tqdm_prog.set_description(f"Epoch {e+1} [{phase.upper()}]: Loss: {loss/dats:.4f}")

            epoch_loss = loss / dats
            losses[phase].append(epoch_loss)

            if phase == "val" and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_weight = copy.deepcopy(model.state_dict())
                if savepath:
                    with open(savepath, 'wb') as filehandler:
                        pickle.dump({
                            'best_weight': best_weight,
                            'best_loss': best_loss,
                            'losses': losses,
                            'optimizer': optimizer,
                            'iter_num': iter_num
                        }, filehandler)
                print(f"Best loss found: {best_loss:3.4f}")
        
        
        x = torch.tensor(tokenizer.encode("he is a"), dtype=torch.int).unsqueeze(0)
        model.eval()
        print("Inference:", ''.join(tokenizer.decode(model.generate(x, max_new_tokens=500, 
                                                                    top_k=2).detach()[0].tolist())))

        eps = list(range(1, len(losses['train'])+1))
        fig = plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.plot(eps, losses['train'], label='train', linestyle='dashed', 
                 color='tab:red')
        plt.plot(eps, losses['val'], label='val', color='tab:red')
        plt.legend(loc="upper left")
        plt.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.grid()
        plt.savefig(os.path.join(
            os.path.dirname(savepath),
            'log.jpg'
        ))
        plt.close(fig)

    print(f'Best loss: {best_loss:3.4f}')
    model.load_state_dict(best_weight)

# %%
train_fn(model, 
         200000,
         optimizer,
         os.path.join('..', 'logs', 'log.pkl'),
         'cpu')

# %%
savepath = os.path.join(
    "..", "logs", "log.pkl"
)

def plot_graph(savepath):
    with open(savepath, 'rb') as filehandler:
        prev_train = CPU_Unpickler(filehandler).load()
        best_weight = prev_train['best_weight']
        model.load_state_dict(prev_train['best_weight'])
        losses = prev_train['losses']
        best_loss = prev_train['best_loss']
        print(f"Loaded model with loss: {best_loss:0.4f}")

    epochs = range(1, len(losses['train'])+1)
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(epochs, losses['train'], color=color, linestyle='dashed', label='train')
    ax1.plot(epochs, losses['val'], color=color, label='val')
    ax1.tick_params(axis='y', labelcolor=color)
    
    
    plt.legend(loc="upper left")
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.grid()
    plt.show()

plot_graph(savepath)

# %%
x = torch.tensor(tokenizer.encode("he is a"), dtype=torch.int).unsqueeze(0)
model.eval()
# model.generate(self, idx, max_new_tokens, temperature=1.0, top_k=None)
# %timeit ''.join(tokenizer.decode(model.generate(x, max_new_tokens=512, top_k=2).detach()[0].tolist()))

# %%
model.eval()
''.join(tokenizer.decode(model.generate(x, max_new_tokens=500, top_k=2).detach()[0].tolist()))

# %%


