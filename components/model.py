import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect
import copy
from dataclasses import dataclass


@dataclass
class GPTConfig:
    # Model parameters: 6.35M [Comparable with MobileNets]
    # Improves large-sequence word generation
    context_len: int = 128
    # Using charachter-level tokenization
    # Actual vocab-size is 75, saving some tokens for future use
    vocab_size: int = 88
    # Improves overall understanding of text
    n_layer: int = 8
    # Heads gives better understanding of word relation
    # Avoids common grammar-level mistakes
    n_head: int = 8
    # Incresing n_embd gave better word memorization
    n_embd: int = 256
    # Regularization
    # [0.0 for now as we want the model to overfit]
    dropout: float = 0.0
    bias: bool = True


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, int(4 * config.n_embd), bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(int(4 * config.n_embd), config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


# mlp = MLP(GPTConfig)
# print(mlp(torch.rand(2, 2, 64)).shape)
# del mlp


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(self, x):
        # batch size, sequence length, embedding dimensionality (n_embd)
        (B, T, C) = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # efficient attention using Flash Attention CUDA kernels
        # y.shape (B, nh, T, hs)
        y = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0,
            is_causal=True,
        )

        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


# catt = CausalSelfAttention(GPTConfig)
# print(catt(torch.rand(2, 2, 64)).shape)
# del catt


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


# block = Block(GPTConfig)
# print(block(torch.rand(2, 2, 64)).shape)
# del block


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = copy.deepcopy(config)
        # preserving one token for previous context

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(self.config.vocab_size, self.config.n_embd),
                # preserving one extra context position for previous context embedding
                wpe=nn.Embedding(self.config.context_len + 1, self.config.n_embd),
                drop=nn.Dropout(self.config.dropout),
                h=nn.ModuleList([Block(self.config) for _ in range(self.config.n_layer)]),
                ln_f=nn.LayerNorm(self.config.n_embd),
            )
        )

        self.lm_head = nn.Linear(self.config.n_embd, self.config.vocab_size, bias=False)
        # https://paperswithcode.com/method/weight-tying
        self.transformer.wte.weight = self.lm_head.weight

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / ((2 * self.config.n_layer) ** 0.5)
                )
        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                 
    def forwardV2(self, idx, targets=None):
        """
        Encodes context as long as it is big
        Slide a window of size context_len over input
        Add the mean of previous context_len embdding
        """
        b, t = idx.size()
        device = idx.device
        p = 0
        context_embd = None
        
        if t <= self.config.context_len:
            return self.forward(idx, targets=targets)
        
        while p < t:
            window = None
            if context_embd is not None:
                # print("Window:", p, p + self.config.context_len, t)
                window = idx[:, p : p + self.config.context_len]
                p = p + self.config.context_len
                
                tok_emb = self.transformer.wte(window)
                pos = torch.arange(0, window.size()[1] + 1, dtype=torch.long, device=device)
                pos_emb = self.transformer.wpe(pos)
                tok_emb = torch.cat((context_embd, tok_emb), dim=1)
                x = self.transformer.drop(tok_emb + pos_emb)
                
                # only proceed computing logits if this is the final window
                if p < t:
                    continue
                
                if targets is not None:
                    # if we are given some desired targets also calculate the loss
                    logits = self.lm_head(x)
                    logits = logits[:, 1:, :]
                    targets = targets[:, -logits.size()[1]:]
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        targets.reshape(-1),
                        ignore_index=-1,
                    )
                else:
                    # inference-time mini-optimization: only forward the lm_head on the very last position
                    logits = self.lm_head(
                        x[:, [-1], :]
                    )  # note: using list [-1] to preserve the time dim
                    loss = None
                    
                return logits, loss

            else:
                # print("*Window:", p, p + self.config.context_len)
                window = idx[:, p : p + self.config.context_len]
                p = p + self.config.context_len 
                context_embd = self.forward(window, embedding=True).mean(dim=1, keepdim=True)
            
            
    def forward(self, idx, targets=None, label_smoothing=None, embedding=False):
        device = idx.device
        b, t = idx.size()
        assert (
            t <= self.config.context_len
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.context_len}"
        # 0'th position is reserved for previous context embedding
        pos = torch.arange(1, t+1, dtype=torch.long, device=device)  # shape (t)
        # forward the GPT model itself
        # token embeddings of shape (b, t, n_embd)
        tok_emb = self.transformer.wte(idx)
        # position embeddings of shape (t, n_embd)
        pos_emb = self.transformer.wpe(pos)
        # add position embeddings
        x = self.transformer.drop(tok_emb + pos_emb)

        # propagating through transformers
        for i, block in enumerate(self.transformer.h):
            x = block(x)
        x = self.transformer.ln_f(x)
        
        # return the embeddings if required
        if embedding:
            return x

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(
                x[:, [-1], :]
            )  # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at context_len
            idx_cond = (
                idx
                if idx.size(1) <= self.config.context_len
                else idx[:, -self.config.context_len :]
            )
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
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

        return idx



if __name__ == "__main__":
    GPTConfig.context_len = 4
    print("Input len", GPTConfig.context_len*5)
    gpt = GPT(GPTConfig)
    inp = torch.ones((2, GPTConfig.context_len*5), dtype=torch.long)
    # out, loss = gpt(inp)
    
    print("Final out", gpt.forwardV2(inp)[0].shape)
    
    #print(out.shape, loss)
    #del gpt, out, loss
