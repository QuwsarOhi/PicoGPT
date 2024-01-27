import torch
import torch.nn as nn
from dataclasses import dataclass
import os
import io
import pickle
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
from components.model import GPT, GPTConfig
from components.tokenizer import Tokenizer
from components.dataloader import TinyShakespere, WikiData, TinyTextBook, OpenOrca
from argparse import ArgumentParser

# Argument parsing
parser = ArgumentParser()
parser.add_argument("--init_weight", type=str, default=None)
parser.add_argument("--fix_lr", type=bool, default=True)
parser.add_argument("--dataset", type=str, default="TinyTextBook")
args, leftovers = parser.parse_known_args()

@dataclass
class TrainConfig:
    batch_size: int = 8
    dtype: str = (
        "bfloat16"
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else "float16"
    )
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    warmup_iters = 2000
    learning_rate = 6e-4
    lr_decay_iters = 600000
    min_lr = 6e-5
    weight_decay = 1e-1
    beta1 = 0.9
    beta2 = 0.95
    label_smoothing = 0.0
    fixed_learning_rate = 3e-5 if args.fix_lr else None


def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < TrainConfig.warmup_iters:
        return TrainConfig.learning_rate * it / TrainConfig.warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > TrainConfig.lr_decay_iters:
        return TrainConfig.min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - TrainConfig.warmup_iters) / (
        TrainConfig.lr_decay_iters - TrainConfig.warmup_iters
    )
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return TrainConfig.min_lr + coeff * (TrainConfig.learning_rate - TrainConfig.min_lr)


# https://github.com/pytorch/pytorch/issues/16797#issuecomment-633423219
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        else:
            return super().find_class(module, name)


def train_fn(
    model: nn.Module,
    epoch: int,
    optimizer: torch.optim.Optimizer,
    savepath: str = None,
):
    best_loss = float("inf")
    iter_num = 0
    train_phases = ["train", "val"]
    losses = {phase: [] for phase in train_phases}

    if savepath and os.path.exists(savepath):
        # Try loading the model and weight
        try:
            with open(savepath, "rb") as filehandler:
                prev_train = CPU_Unpickler(filehandler).load()
                best_weight = prev_train["best_weight"]
                model.load_state_dict(best_weight, strict=False)
                losses = prev_train["losses"]
                best_loss = prev_train["best_loss"]
                optimizer = prev_train["optimizer"]
                iter_num = prev_train["iter_num"]
                print(f"Loaded model with loss: {best_loss:0.4f}")
        except Exception as e:
            print(f"Could not load from path: {savepath}\n", repr(e))

    for e in range(epoch):
        for phase in train_phases:
            is_training = phase == "train"
            model.train() if is_training else model.eval()
            loss, dats = (
                0.0,
                0.0,
            )
            tqdm_prog = tqdm(range(500))

            for _ in tqdm_prog:
                x, y = data.get_batch(phase, TrainConfig.batch_size, TrainConfig.device)

                with torch.set_grad_enabled(phase == "train"):
                    _, batch_loss = model(
                        x,
                        y,
                        label_smoothing=TrainConfig.label_smoothing
                        if is_training
                        else 0.0,
                    )

                    if is_training:
                        batch_loss.backward()
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        # LR scheduler
                        # only update lr when it is not fixed
                        if TrainConfig.fixed_learning_rate is None:
                            lr = get_lr(iter_num)
                            for param_group in optimizer.param_groups:
                                param_group["lr"] = lr
                        else:
                            lr = TrainConfig.fixed_learning_rate
                        iter_num += 1

                # Stats
                dats += x.size(0)
                loss += batch_loss.item() * x.size(0)
                tqdm_prog.set_description(
                    f"Epoch {e+1} [{phase.upper()}]: Loss: {loss/dats:.4f}, lr: {lr:0.6f}"
                )

            epoch_loss = loss / dats
            losses[phase].append(epoch_loss)

        # Save training state
        if losses["val"][-1] < best_loss:
            best_loss = losses["val"][-1]
            print(f"Best loss found: {best_loss:3.4f}")

        if savepath:
            with open(savepath, "wb") as filehandler:
                pickle.dump(
                    {
                        "best_weight": model.state_dict(),
                        "best_loss": best_loss,
                        "losses": losses,
                        "optimizer": optimizer,
                        "iter_num": iter_num,
                    },
                    filehandler,
                )

        # Inference test
        model.eval()
        x = torch.tensor(tokenizer.encode("He is a"), dtype=torch.int,
                         device=TrainConfig.device).unsqueeze(0)
        print(
            "Inference:",
            "".join(
                tokenizer.decode(
                    model.generate(x, max_new_tokens=500, temperature=0.5).detach()[0].tolist()
                )
            ),
        )

        eps = list(range(1, len(losses["train"]) + 1))
        fig = plt.figure()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.plot(
            eps, losses["train"], label="train", linestyle="dashed", color="tab:red"
        )
        plt.plot(eps, losses["val"], label="val", color="tab:red")
        plt.legend(loc="upper left")
        plt.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.grid()
        plt.savefig(os.path.join(os.path.dirname(savepath), "log.jpg"))
        plt.close(fig)

    print(f"Best loss: {best_loss:3.4f}")


if __name__ == "__main__":
    # Loading dataset
    tokenizer = Tokenizer()

    if args.dataset == "WikiData":
        data = WikiData(tokenizer)
    elif args.dataset == "TinyShakespere":
        data = TinyShakespere(tokenizer)
    elif args.dataset == "TinyTextBook":
        data = TinyTextBook(tokenizer)
    elif args.dataset == "OpenOrca":
        data = OpenOrca(tokenizer)
    else:
        raise ValueError(f"Invalid dataset name {args.dataset}")
    print(f"Training on {args.dataset} dataset")

    model = GPT(GPTConfig).to(TrainConfig.device)
    optimizer = model.configure_optimizers(
        TrainConfig.weight_decay,
        TrainConfig.fixed_learning_rate if TrainConfig.fixed_learning_rate else TrainConfig.learning_rate,
        (TrainConfig.beta1, TrainConfig.beta2),
        TrainConfig.device,
    )
    
    if args.init_weight is not None:
        init_path = os.path.join('.', 'logs', args.init_weight, 'log.pkl')
        with open(init_path, "rb") as filehandler:
            prev_train = CPU_Unpickler(filehandler).load()
            best_weight = prev_train["best_weight"]
            model.load_state_dict(best_weight, strict=True)
        print(f"Loaded weight form {init_path}")
        
    train_fn(model, 300, optimizer, os.path.join(".", "logs", args.dataset.lower(), "log.pkl"))
