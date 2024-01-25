# PicoGPT
This project implements a no-fuss GPT model that avoids fancy multi-GPU training strategies. The GPT contains 6.34 million parameters (very similar to MobileNets) and trained in an old hardware having Intel-i3 processor with 8 Gigs of ram. The model implementation is inherited from karapathy's famous [nanoGPT](https://github.com/karpathy/nanoGPT) implementation with some of my own basic modifications.

The model is trained on ~20 Gigs [Wikipedia dataset](https://huggingface.co/datasets/wikipedia) from Huggingface. The model is built and trained for educational purposes.

```
PicoGPT
│
├── dataset
│   └── tinyshakespeare.txt
├── inference.py
├── logs
│   ├── log.jpg
│   └── log.pkl
├── model
│   ├── dataloader.py
│   ├── model.py
│   └── tokenizer.py
├── LICENSE
├── README.md
└── train.py
```

### References:
* https://arxiv.org/abs/1706.03762
* https://github.com/karpathy/nanoGPT
* https://github.com/karpathy/ng-video-lecture/tree/master
