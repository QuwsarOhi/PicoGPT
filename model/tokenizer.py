import string
from typing import List


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

    def __init__(self, specials: List[str] = ["<?>"]) -> None:
        self.enc = dict((c, i) for i, c in enumerate(specials))
        self.dec = dict((i, c) for i, c in enumerate(specials))
        self.specials = specials
        self.__build()

    def __build(self):
        n = len(self.specials)
        for i, c in enumerate(string.printable):
            self.enc[c] = i + n
            self.dec[i + n] = c
        # vocab size is 101
        self.vocab_size = len(self.enc)

    def encode(self, x: List[str]) -> List[int]:
        return [(self.enc[c] if c in self.enc else 0) for c in x]

    def decode(self, x: List[int]) -> List[str]:
        return [(self.dec[i] if i in self.dec else self.dec[0]) for i in x]


class TokenizerV2:
    """
    Simple bi-gram tokenizer.

    This tokenizer only maps printable ASCII charachters to integers.
    There are a total of 100 printable charachters in ASCII table.
    Uppercase charachters are excluded.
    The list of printable charachters are generated from: string.printable

    Special tokens:
    <?> is considered as special token, which is also known as unknown token.
    Otherwise there is no other special tokens.

    Although there are 724 tokens (including <?>) GPT predicts 736 tokens.
    The rest of the 12 tokens are kept for future processing (if required).

    Parameters:
    -----------
        specials: A list of special tokens if required. <?> is a special uknown token.

    """

    # A trick to save memory
    __slots__ = ["vocab_size", "enc", "dec"]

    def __init__(self, specials: List[str] = []) -> None:
        self.__build(specials)

    def __build(self, specials):
        self.vocab_size = 0
        self.enc = {"<?>": 0}
        self.dec = {0: "<?>"}

        for a in string.ascii_lowercase:
            for b in string.ascii_lowercase:
                if a != b:
                    self.enc[a + b] = self.vocab_size
                    self.dec[self.vocab_size] = a + b
                    self.vocab_size += 1

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

    def get_id(self, t):
        if t in self.enc:
            return self.enc[t]
        return 0

    def encode(self, x: List[str]) -> List[int]:
        N = len(x)
        ret = []
        i = 0
        x = x.lower()
        while i < N:
            if i + 1 < N:
                tok = "".join(x[i : i + 2])
                if tok in self.enc:
                    ret.append(self.enc[tok])
                    i += 2
                else:
                    ret.append(self.get_id(x[i]))
                    i += 1
            else:
                ret.append(self.get_id(x[i]))
                i += 1
        return ret

    def decode(self, x: List[int]) -> List[str]:
        return [(self.dec[i] if i in self.dec else self.dec[0]) for i in x]


if __name__ == "__main__":
    tokenizer = TokenizerV2()
    print(tokenizer.vocab_size)
    print(tokenizer.encode("Thiss is a string."))
    print(tokenizer.decode(tokenizer.encode("Thiss is a string.")))
