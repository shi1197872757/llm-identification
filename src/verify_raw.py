"""
Verify the three raw CSVs are aligned before merging.
Checks: row counts, prompt overlap, per-LLM label uniqueness.
"""
import re
import unicodedata
import pandas as pd


def norm(s):
    """Aggressive normalization for comparison only — never applied to model input."""
    s = unicodedata.normalize("NFKC", str(s))
    s = s.replace("\u2018", "'").replace("\u2019", "'")
    s = s.replace("\u201c", '"').replace("\u201d", '"')
    s = s.replace("\u2014", "--").replace("\u2013", "-")
    s = s.replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


files = {
    "ChatGPT":  "data/raw/chatgpt.csv",
    "Gemini":   "data/raw/gemini.csv",
    "DeepSeek": "data/raw/deepseek.csv",
}

dfs = {}
print("=" * 60)
print("RAW FILE INSPECTION")
print("=" * 60)

for name, path in files.items():
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    assert {"input", "output"}.issubset(df.columns), f"{name}: missing input/output"
    df = df.dropna(subset=["input", "output"])
    dfs[name] = df
    print(f"\n{name}:")
    print(f"  rows        = {len(df)}")
    print(f"  unique prompts = {df['input'].nunique()}")
    print(f"  null cells  = {df.isnull().sum().sum()}")
    print(f"  avg output len = {df['output'].str.len().mean():.0f}")

# ---- Prompt alignment ----
print("\n" + "=" * 60)
print("PROMPT ALIGNMENT")
print("=" * 60)

sets_norm = {name: set(df["input"].map(norm)) for name, df in dfs.items()}
all_common = set.intersection(*sets_norm.values())

print(f"ChatGPT ∩ Gemini           = {len(sets_norm['ChatGPT'] & sets_norm['Gemini'])}")
print(f"ChatGPT ∩ DeepSeek         = {len(sets_norm['ChatGPT'] & sets_norm['DeepSeek'])}")
print(f"Gemini  ∩ DeepSeek         = {len(sets_norm['Gemini'] & sets_norm['DeepSeek'])}")
print(f"All three (after normalize)= {len(all_common)}")

# Which prompts are NOT shared?
per_prompt_llms = {}
for name, s in sets_norm.items():
    for p in s:
        per_prompt_llms.setdefault(p, set()).add(name)

counts = pd.Series({p: len(llms) for p, llms in per_prompt_llms.items()})
print("\nPrompts by #LLMs answering them:")
print(counts.value_counts().sort_index())

missing = {p: llms for p, llms in per_prompt_llms.items() if len(llms) < 3}
if missing:
    print(f"\n⚠️  {len(missing)} prompt(s) missing from some LLMs:")
    for p, llms in list(missing.items())[:10]:
        got = sorted(llms)
        print(f"    [{len(llms)}/3] {got}  | {p[:80]}")

# ---- Verdict ----
print("\n" + "=" * 60)
if len(all_common) == 100:
    print("✅ VERDICT: Perfect alignment. 100 shared prompts across all 3 LLMs.")
elif len(all_common) >= 90:
    print(f"⚠️  VERDICT: {len(all_common)}/100 prompts aligned. "
          "Proceed but note the drop in the report.")
else:
    print(f"❌ VERDICT: Only {len(all_common)}/100 prompts aligned. "
          "Fix prompt consistency before training.")
print("=" * 60)