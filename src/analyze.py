"""
Generates confusion matrices for all 6 combos.
Picks an available seed automatically (any file matching models/{model}_{use}_seed*.pt).
"""
import glob
import pickle
import re
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from dataset import TextDataset
from cnn import CNNClassifier
from rnn import RNNClassifier

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
vocab = pickle.load(open("models/vocab.pkl", "rb"))
le = pickle.load(open("models/label_encoder.pkl", "rb"))
classes = list(le.classes_)


def find_checkpoint(model_name, use):
    """Return the path to any existing checkpoint for (model, use), preferring seed=1."""
    pattern = f"models/{model_name}_{use}_seed*.pt"
    matches = glob.glob(pattern)
    if not matches:
        return None
    # Sort by seed number, prefer smallest (seed 1 usually exists after sweep)
    def seed_of(path):
        m = re.search(r"seed(\d+)", path)
        return int(m.group(1)) if m else 9999
    matches.sort(key=seed_of)
    return matches[0]


def load_model(model_name, use, path, max_len=200):
    m = CNNClassifier(len(vocab)) if model_name == "cnn" else RNNClassifier(len(vocab))
    m.load_state_dict(torch.load(path, map_location=DEVICE))
    m.to(DEVICE).eval()
    return m


def predict(model, loader):
    preds, gold = [], []
    with torch.no_grad():
        for x, y in loader:
            preds += model(x.to(DEVICE)).argmax(1).cpu().tolist()
            gold += y.tolist()
    return np.array(gold), np.array(preds)


rows = []
skipped = []

for model_name in ["cnn", "rnn"]:
    for use in ["output", "input", "both"]:
        path = find_checkpoint(model_name, use)
        if path is None:
            skipped.append(f"{model_name}/{use}")
            print(f"⚠️  No checkpoint for {model_name}/{use} — skipping")
            continue

        seed_used = re.search(r"seed(\d+)", path).group(1)
        print(f"\n--- {model_name.upper()} / {use} (using seed {seed_used}) ---")

        ds = TextDataset("data/processed/test.csv", vocab, 200, use)
        dl = DataLoader(ds, batch_size=64)
        m = load_model(model_name, use, path)
        g, p = predict(m, dl)

        cm = confusion_matrix(g, p, labels=[0, 1, 2])
        fig, ax = plt.subplots(figsize=(4.5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=classes, yticklabels=classes, ax=ax)
        ax.set_title(f"{model_name.upper()} — {use}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        plt.tight_layout()
        plt.savefig(f"results/cm_{model_name}_{use}.png", dpi=140)
        plt.close()

        acc = accuracy_score(g, p)
        macro_f1 = classification_report(
            g, p, digits=3, zero_division=0, output_dict=True
        )["macro avg"]["f1-score"]

        rows.append({
            "model": model_name.upper(),
            "feature": use,
            "seed": seed_used,
            "test_acc": round(acc, 3),
            "test_macro_f1": round(macro_f1, 3),
        })

if rows:
    df = pd.DataFrame(rows)
    df.to_csv("results/final_table.csv", index=False)
    print("\n" + "=" * 60)
    print("FINAL TABLE (single seed per combo)")
    print("=" * 60)
    print(df.to_string(index=False))
    print("\nSaved: results/final_table.csv")
    print("Saved 6 PNGs to results/")

if skipped:
    print(f"\n⚠️  Missing checkpoints for: {skipped}")
    print("    Run: python src/run_sweep.py")
else:
    print("\n✅ All 6 confusion matrices generated.")