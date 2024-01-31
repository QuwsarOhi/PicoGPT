import torch
import os
import torch.nn.functional as F
from components.model import GPT, GPTConfig, CPU_Unpickler
from components.tokenizer import Tokenizer
from argparse import ArgumentParser


# Argument parsing
parser = ArgumentParser()
parser.add_argument("--weight", type=str, default="wikidata_ct1280")
parser.add_argument("--chat", type=bool, default=False)
args, leftovers = parser.parse_known_args()

model = GPT(GPTConfig)
tokenizer = Tokenizer()

savepath = os.path.join(".", "logs", args.weight, "log.pkl")
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
        # forward the model to get the logits for the index in the sequence
        logits, _ = model(idx)
        # pluck the logits at the final step and scale by desired temperature
        logits = logits[:, -1, :] / temperature

        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")

        # apply softmax to convert logits to (normalized) probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1)
        # append sampled index to the running sequence and continue
        idx = torch.cat((idx, idx_next), dim=1)
        tok = tokenizer.decode(idx_next[0].tolist())[0]
        print(tok, end="")
        if tok == ".":
            print()
            return
    print("\n<STRIPPED>\n")


while True:
    print("Input: ", end="")
    x = input()
    x = tokenizer.encode(x)

    if args.chat:
        x = (
            # [76]
            # + tokenizer.encode(
            #     "You are an AI assistant. You will be given a task. You must generate a detailed and long answer."
            # )
            # + [77]
            # + tokenizer.encode("\n")
            [78] + x + [79] + tokenizer.encode("\n") + [80]
        )

    x = torch.tensor(x, dtype=torch.long).unsqueeze(0)
    generate(x, max_new_tokens=256, temperature=0.5, top_k=3)
