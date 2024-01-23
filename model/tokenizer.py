import string
from typing import List
import pickle
import torch


class TokenizerV1:
    """
    Very simple charachter-level tokenizer.

    This tokenizer only maps printable ASCII charachters to integers.
    There are a total of 100 printable charachters in ASCII table.
    The list of printable charachters are generated from: string.printable

    Special tokens:
    <?> is considered as special token, which is also known as unknown token.
    Otherwise there is no other special tokens.

    Although there are 101 tokens (including <?>) GPT predicts 128 tokens.
    The rest of the tokens are kept for future processing (if required).

    Parameters:
    -----------
        specials: A list of special tokens if required. <?> is a special uknown token.

    """

    # A trick to save memory
    __slots__ = ["vocab_size", "enc", "dec", "specials", "weight"]

    def __init__(self, specials: List[str] = ["<?>"], weight_path: str = None) -> None:
        self.enc = dict((c, i) for i, c in enumerate(specials))
        self.dec = dict((i, c) for i, c in enumerate(specials))
        self.specials = specials
        self.__build_weight(weight_path)
        self.__build()

    def __build(self):
        n = len(self.specials)
        for i, c in enumerate(string.printable):
            self.enc[c] = i + n
            self.dec[i + n] = c
        # vocab size is 101
        self.vocab_size = len(self.enc)

    def __build_weight(self, weight_path=None):
        """
        This weight is specifically for crossentropy loss to add bias to certain charachters
        """
        self.weight = None
        if weight_path is None:
            return

        with open(weight_path, "rb") as filehandler:
            self.weight = list(pickle.load(filehandler).values())

        # As the vocab for GPT is greater than the tokenizer vocab, pad the remaining
        while len(self.weight) < 128:
            self.weight.append(0.0)

        self.weight = torch.tensor(self.weight, dtype=torch.float)

    def encode(self, x: List[str]) -> List[int]:
        return [(self.enc[c] if c in self.enc else 0) for c in x]

    def decode(self, x: List[int]) -> List[str]:
        return [(self.dec[i] if i in self.dec else self.dec[0]) for i in x]
