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

while True:
    # conv = (
    #     [76]
    #     + tokenizer.encode(
    #         "You are an AI assistant. You will be given a task. You must generate a detailed and long answer."
    #     )
    #     + [77]
    #     + tokenizer.encode("\n")
    # )
    conv = []
    
    while True:
        print("Input: ", end="")
        ques = input()
        ques = tokenizer.encode(ques)
        ques = [78] + ques + [79] + tokenizer.encode("\n") + [80]

        # Add the question with the previous conversation so that GPT knows
        # what happened before
        conv = conv + ques
        conv_len = len(conv)
        
        # print("-+-+"*20)
        # print("".join(tokenizer.decode(conv)))
        # print("-+-+"*20)
        
        ques = torch.tensor(conv, dtype=torch.long).unsqueeze(0)
        ans = model.generate(ques, max_new_tokens=256, temperature=0.5).detach()[0].tolist()
        decoded_ans = "".join(tokenizer.decode(ans[conv_len:]))
        print(decoded_ans)
        # Add the answer to the conversation
        conv = ans + [81] + tokenizer.encode("\n")
