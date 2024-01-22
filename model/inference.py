import torch
import pickle
import io
import os
import torch.nn.functional as F
from model import GPT, GPTConfig
from tokenizer import Tokenizer


model = GPT(GPTConfig)
tokenizer = Tokenizer()


# https://github.com/pytorch/pytorch/issues/16797#issuecomment-633423219
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        else:
            return super().find_class(module, name)


savepath = os.path.join("..", "logs", "log.pkl")
with open(savepath, "rb") as filehandler:
    model.load_state_dict(CPU_Unpickler(filehandler).load()["best_weight"])
model.eval()


@torch.inference_mode
def generate(idx, max_new_tokens, temperature=1.0, top_k=None):
    """
    Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
    the sequence max_new_tokens times, feeding the predictions back into the model each time.
    Most likely you'll want to make sure to be in model.eval() mode of operation for this.
    """
    for _ in range(max_new_tokens):
        # if the sequence context is growing too long we must crop it at context_len
        idx_cond = (
            idx
            if idx.size(1) <= GPTConfig.context_len
            else idx[:, -GPTConfig.context_len :]
        )
        # forward the model to get the logits for the index in the sequence
        logits, _ = model(idx_cond)
        # pluck the logits at the final step and scale by desired temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop the logits to only the top k options
        # if top_k is not None and top_k == 1:
        #     v, idx_next = torch.topk(logits, min(top_k, logits.size(-1)))
        #     idx = torch.cat((idx, idx_next), dim=1)
        #     print(tokenizer.decode(idx_next[0].tolist())[0], end='')
        #     continue

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")

        # apply softmax to convert logits to (normalized) probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1)
        # append sampled index to the running sequence and continue
        idx = torch.cat((idx, idx_next), dim=1)
        print(tokenizer.decode(idx_next[0].tolist())[0], end="")


while True:
    print("Input: ", end="")
    model.eval()
    x = input()
    x = torch.tensor(tokenizer.encode(x), dtype=torch.int).unsqueeze(0)

    with torch.inference_mode():
        generate(x, max_new_tokens=500, top_k=5)
        print("\nEnded\n")
