from collections import Counter
import re
import pickle
import pandas as pd


TOKEN_RE = re.compile(r"[a-z0-9']+|[.,!?;:()\[\]{}*#`_\-]")


def tokenize(text):
    return TOKEN_RE.findall(str(text).lower())


def build_vocab(texts, max_size=20000, min_freq=2):
    counter = Counter()
    for t in texts:
        counter.update(tokenize(t))
    vocab = {"<pad>": 0, "<unk>": 1}
    for w, c in counter.most_common():
        if c < min_freq or len(vocab) >= max_size:
            break
        vocab[w] = len(vocab)
    return vocab


if __name__ == "__main__":
    train = pd.read_csv("data/processed/train.csv")
    all_text = pd.concat([
        train["input_clean"].astype(str),
        train["output_clean"].astype(str),
    ]).tolist()
    vocab = build_vocab(all_text)
    pickle.dump(vocab, open("models/vocab.pkl", "wb"))
    print("Vocab size:", len(vocab))

    # Diagnostics
    counter = Counter()
    for t in all_text:
        counter.update(tokenize(t))
    total = sum(counter.values())
    unk_tokens = sum(c for w, c in counter.items() if w not in vocab)
    print(f"Total training tokens: {total:,}")
    print(f"Unique token types:    {len(counter):,}")
    print(f"Kept in vocab:         {len(vocab) - 2:,}")
    print(f"OOV token rate:        {unk_tokens/total*100:.2f}%")