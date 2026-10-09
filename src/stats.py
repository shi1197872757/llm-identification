import re
import pandas as pd
from collections import Counter
from vocab import tokenize

df = pd.read_csv("data/processed/train.csv")
df["length"] = df["output_clean"].str.len()

print("=" * 60)
print("RESPONSE LENGTH (chars) BY LLM")
print("=" * 60)
print(df.groupby("llm")["length"].describe()[["mean", "std", "min", "max"]].round(1))


print("\n" + "=" * 60)
print("WORD-LEVEL STATS BY LLM")
print("=" * 60)
for llm, g in df.groupby("llm"):
    words = []
    for t in g["output_clean"]:
        words += tokenize(str(t))
    c = Counter(words)
    total = sum(c.values())
    print(f"{llm:9s}  tokens={total:6d}  unique={len(c):5d}  "
          f"TTR={len(c)/total:.3f}")


print("\n" + "=" * 60)
print("MARKDOWN / LATEX MARKER FREQUENCY (mean per doc)")
print("=" * 60)


def count_literal(text, marker):
    """Literal substring count — no regex interpretation."""
    if not isinstance(text, str):
        return 0
    return text.count(marker)


markers = [
    ("```",  "code fence"),
    ("`",    "inline code"),
    ("**",   "bold"),
    ("__",   "bold alt"),
    ("$$",   "latex display"),
    ("$",    "latex"),
    ("|",    "table pipe"),
    ("![",   "image"),
    ("###",  "h3"),
    ("##",   "h2"),
    ("# ",   "h1"),
    ("- ",   "bullet dash"),
    ("1. ",  "numbered list"),
]

results = []
for marker, name in markers:
    row = {"marker": name, "pattern": marker}
    for llm, g in df.groupby("llm"):
        mean = g["output_clean"].apply(lambda s: count_literal(s, marker)).mean()
        row[llm] = round(mean, 2)
    results.append(row)

marker_df = pd.DataFrame(results)
print(marker_df.to_string(index=False))


print("\nCompact view:")
for _, r in marker_df.iterrows():
    d = {llm: r[llm] for llm in ["ChatGPT", "DeepSeek", "Gemini"]}
    print(f"  {r['marker']:14s} : {d}")


print("\n" + "=" * 60)
print("NORMALIZED MARKER DENSITY (per 1000 characters)")
print("=" * 60)
for _, r in marker_df.iterrows():
    line = f"  {r['marker']:14s} :"
    for llm in ["ChatGPT", "DeepSeek", "Gemini"]:
        g = df[df["llm"] == llm]
        total_markers = g["output_clean"].apply(lambda s: count_literal(s, r["pattern"])).sum()
        total_chars = g["output_clean"].str.len().sum()
        density = total_markers / total_chars * 1000
        line += f"  {llm}={density:5.2f}"
    print(line)