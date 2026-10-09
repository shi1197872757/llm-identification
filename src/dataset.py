import torch
import pandas as pd
from torch.utils.data import Dataset
from vocab import tokenize


class TextDataset(Dataset):
    """use: 'output' | 'input' | 'both'"""

    def __init__(self, csv, vocab, max_len=200, use="output"):
        self.df = pd.read_csv(csv)
        self.vocab = vocab
        self.max_len = max_len
        self.use = use

    def __len__(self):
        return len(self.df)

    def _encode(self, text):
        ids = [self.vocab.get(w, 1) for w in tokenize(text)]
        ids = ids[:self.max_len]
        ids += [0] * (self.max_len - len(ids))
        return ids

    def __getitem__(self, i):
        r = self.df.iloc[i]
        if self.use == "output":
            text = r["output_clean"]
        elif self.use == "input":
            text = r["input_clean"]
        else:
            text = str(r["input_clean"]) + " [SEP] " + str(r["output_clean"])
        x = torch.tensor(self._encode(text), dtype=torch.long)
        y = torch.tensor(int(r["label"]), dtype=torch.long)
        return x, y