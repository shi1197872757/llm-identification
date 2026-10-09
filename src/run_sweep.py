"""
Runs all 6 combos across 5 seeds, saves mean ± std to results/sweep_results.csv.
"""
import numpy as np
import pandas as pd
from train import run

COMBOS = [
    ("cnn", "output"),
    ("rnn", "output"),
    ("cnn", "input"),
    ("rnn", "input"),
    ("cnn", "both"),
    ("rnn", "both"),
]
SEEDS = [1, 2, 3, 4, 5]

rows = []
for model, use in COMBOS:
    for seed in SEEDS:
        r = run(model, use, epochs=8, seed=seed, verbose=True)
        rows.append({
            "model": model,
            "use": use,
            "seed": seed,
            "test_acc": r["test_acc"],
            "test_macro_f1": r["test_f1"],
            "best_val_f1": r["best_val_f1"],
        })

df = pd.DataFrame(rows)
df.to_csv("results/sweep_raw.csv", index=False)

summary = (
    df.groupby(["model", "use"])
    .agg(
        acc_mean=("test_acc", "mean"),
        acc_std=("test_acc", "std"),
        f1_mean=("test_macro_f1", "mean"),
        f1_std=("test_macro_f1", "std"),
    )
    .round(3)
)
print("\n" + "=" * 60)
print("5-SEED SWEEP SUMMARY")
print("=" * 60)
print(summary)
summary.to_csv("results/sweep_results.csv")

# Sanity checks
for model in ["cnn", "rnn"]:
    m = summary.loc[(model, "input"), "acc_mean"]
    if m > 0.45:
        print(f"\n⚠️  {model}/input acc_mean={m:.3f} is above 0.45. "
              "Check raw prompt alignment (verify_raw.py).")
    elif m < 0.20:
        print(f"\n⚠️  {model}/input acc_mean={m:.3f} is well below 0.33. "
              "Model instability — check gradient clipping / LR.")
    else:
        print(f"\n✅ {model}/input acc_mean={m:.3f} — consistent with chance.")