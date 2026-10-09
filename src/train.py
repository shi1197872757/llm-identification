import argparse
import pickle
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
)
from dataset import TextDataset
from cnn import CNNClassifier
from rnn import RNNClassifier


def get_model(name, vocab_size):
    return CNNClassifier(vocab_size) if name == "cnn" else RNNClassifier(vocab_size)


def evaluate(model, loader, device):
    model.eval()
    preds, gold = [], []
    with torch.no_grad():
        for x, y in loader:
            p = model(x.to(device)).argmax(1).cpu().numpy()
            preds += p.tolist()
            gold += y.tolist()
    return np.array(gold), np.array(preds)


def run(model_name, use, epochs=8, batch=16, lr=1e-3, max_len=200,
        seed=42, weight_decay=1e-4, dropout=0.4, verbose=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vocab = pickle.load(open("models/vocab.pkl", "rb"))
    train_ds = TextDataset("data/processed/train.csv", vocab, max_len, use)
    val_ds   = TextDataset("data/processed/val.csv",   vocab, max_len, use)
    test_ds  = TextDataset("data/processed/test.csv",  vocab, max_len, use)

    tr = DataLoader(train_ds, batch_size=batch, shuffle=True)
    va = DataLoader(val_ds,   batch_size=batch)
    te = DataLoader(test_ds,  batch_size=batch)

    model = get_model(model_name, len(vocab)).to(device)
    # Set dropout if model supports it (they both do)
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = dropout

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()

    best_f1, best_state, patience, wait = -1, None, 4, 0
    history = []

    for ep in range(epochs):
        model.train()
        total = 0
        t0 = time.time()
        for x, y in tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            total += loss.item() * x.size(0)

        g, p = evaluate(model, va, device)
        f1 = f1_score(g, p, average="macro", zero_division=0)
        acc = accuracy_score(g, p)
        history.append({"epoch": ep + 1, "loss": total / len(train_ds),
                        "val_acc": acc, "val_f1": f1})

        if verbose:
            print(f"[{model_name}/{use}/seed{seed}] ep{ep+1} "
                  f"loss={history[-1]['loss']:.3f} "
                  f"val_acc={acc:.3f} val_f1={f1:.3f} "
                  f"({time.time()-t0:.1f}s)")

        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    print("Early stop.")
                break

    if best_state is None:
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)

    g, p = evaluate(model, te, device)
    report = classification_report(g, p, digits=3, zero_division=0)
    cm = confusion_matrix(g, p)
    test_acc = accuracy_score(g, p)
    test_f1 = f1_score(g, p, average="macro", zero_division=0)

    if verbose:
        print("\n=== TEST ===")
        print(report)
        print("Confusion matrix:\n", cm)

    torch.save(best_state, f"models/{model_name}_{use}_seed{seed}.pt")

    return {
        "model": model_name, "use": use, "seed": seed,
        "test_acc": test_acc, "test_f1": test_f1,
        "best_val_f1": best_f1,
        "confusion_matrix": cm,
        "report": report,
        "history": history,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=["cnn", "rnn"])
    ap.add_argument("use",   choices=["output", "input", "both"])
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch",  type=int, default=16)
    ap.add_argument("--lr",     type=float, default=1e-3)
    ap.add_argument("--max_len", type=int, default=200)
    ap.add_argument("--seed",   type=int, default=42)
    ap.add_argument("--dropout", type=float, default=0.4)
    ap.add_argument("--wd",     type=float, default=1e-4)
    a = ap.parse_args()
    run(a.model, a.use, a.epochs, a.batch, a.lr, a.max_len,
        a.seed, a.wd, a.dropout)