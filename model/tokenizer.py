import string
from typing import List


class Tokenizer:
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
    
    vocab size is 75
    """

    # A trick to save memory
    __slots__ = ["vocab_size", "enc", "dec"]

    def __init__(self, specials: List[str] = []) -> None:
        self.__build(specials)

    def __build(self, specials):
        self.vocab_size = 1
        self.enc = {"<?>": 0}
        self.dec = {0: "<?>"}

        for c in string.printable:
            if c.isupper():
                continue
            self.enc[c] = self.vocab_size
            self.dec[self.vocab_size] = c
            self.vocab_size += 1

        for c in specials:
            self.enc[c] = self.vocab_size
            self.dec[self.vocab_size] = c
            self.vocab_size += 1

    def encode(self, x: List[str]) -> List[int]:
        return [(self.enc[c.lower()] if c.lower() in self.enc else 0) for c in x]

    def decode(self, x: List[int]) -> List[str]:
        return [(self.dec[i] if i in self.dec else self.dec[0]) for i in x]


if __name__ == "__main__":
    tokenizer = Tokenizer()
    print(tokenizer.vocab_size)
    print(tokenizer.encode("Thiss is 𐌀 string."))
    print(tokenizer.decode(tokenizer.encode("Thiss is 𐌀 string.")))
