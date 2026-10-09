"""
Full preprocessing pipeline:
  1. Load raw CSVs
  2. Normalize labels
  3. Strip platform widgets (ChatGPT images, Gemini YouTube)
  4. Enforce global prompt alignment (keep only prompts answered by all 3 LLMs)
  5. Balance classes
  6. SPLIT BY PROMPT (not by row) so every split has all 3 LLMs per prompt
  7. Leakage sanity check
"""
import os
import re
import unicodedata
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder

RAW = {
    "ChatGPT":  "data/raw/chatgpt.csv",
    "Gemini":   "data/raw/gemini.csv",
    "DeepSeek": "data/raw/deepseek.csv",
}
OUT_DIR = "data/processed"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------
def norm_key(s):
    """Normalization used ONLY to group prompts across files/splits."""
    s = unicodedata.normalize("NFKC", str(s))
    s = s.replace("\u2018", "'").replace("\u2019", "'")
    s = s.replace("\u201c", '"').replace("\u201d", '"')
    s = s.replace("\u2014", "--").replace("\u2013", "-").replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2014", "--").replace("\u2013", "-").replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


CHATGPT_JUNK = [
    r"^\s*Add to Favorites\s*$",
    r"^\s*Give feedback\s*$",
    r"^\s*Choose .* to .*\.\s*$",
    r"^\s*Move the .* slider .*\.\s*$",
    r"^\s*[a-zA-Z]\s*=\s*[\d.]+\s*.*$",
    r"^\s*\.\.\.\s*$",
    r"^\s*(Writing|Python|Run|Loading)\s*$",
]

def strip_chatgpt_widgets(text):
    if not isinstance(text, str):
        return text
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    text = re.sub(r"https?://\S*images\.openai\.com\S*", "", text)
    text = re.sub(r"https?://\S*cdn\.openai\.com\S*", "", text)
    lines = [l for l in text.split("\n")
             if not any(re.match(p, l) for p in CHATGPT_JUNK)]
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


GEMINI_JUNK = [
    r"^\s*[A-Z][A-Za-z ]{2,40}\s*\+\s*\d+\s*$",
    r"^\s*History Hit\s*$",
    r"^\s*World History Encyclopedia.*$",
]

def strip_gemini_widgets(text):
    if not isinstance(text, str):
        return text
    text = re.sub(r"\[[^\]]*\]\(https?://(www\.)?youtube\.com[^)]*\)", "", text)
    text = re.sub(r"\[[^\]]*\]\(https?://youtu\.be[^)]*\)", "", text)
    text = re.sub(r"https?://(www\.)?youtube\.com/\S*", "", text)
    text = re.sub(r"https?://youtu\.be/\S*", "", text)
    lines = [l for l in text.split("\n")
             if not any(re.match(p, l) for p in GEMINI_JUNK)]
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


# ---------------------------------------------------------------
# 1. Load & normalize labels
# ---------------------------------------------------------------
print("=" * 60)
print("LOADING RAW DATA")
print("=" * 60)

rows = []
for label, path in RAW.items():
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.dropna(subset=["input", "output"]).copy()
    df["llm"] = label
    rows.append(df[["llm", "input", "output"]])
    print(f"  {label:9s}: {len(df)} rows")

df = pd.concat(rows, ignore_index=True)
print(f"Combined: {len(df)} rows")
print(df["llm"].value_counts())


# ---------------------------------------------------------------
# 2. Clean text
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("CLEANING TEXT")
print("=" * 60)

df["input_clean"]  = df["input"].map(clean_text)
df["output_clean"] = df["output"].map(clean_text)

mask = df["llm"] == "ChatGPT"
df.loc[mask, "output_clean"] = df.loc[mask, "output_clean"].map(strip_chatgpt_widgets)
mask = df["llm"] == "Gemini"
df.loc[mask, "output_clean"] = df.loc[mask, "output_clean"].map(strip_gemini_widgets)

before = len(df)
df = df[df["output_clean"].str.len() > 20]
print(f"Dropped {before - len(df)} rows with empty/short output")
print(f"Remaining: {len(df)}")


# ---------------------------------------------------------------
# 3. Enforce global prompt alignment
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("ENFORCING PROMPT ALIGNMENT (global)")
print("=" * 60)

df["prompt_id"] = df["input_clean"].map(norm_key)
llms_per_prompt = df.groupby("prompt_id")["llm"].nunique()
good_prompts = llms_per_prompt[llms_per_prompt == 3].index

print(f"Prompts with all 3 LLMs: {len(good_prompts)}")
kept = df[df["prompt_id"].isin(good_prompts)].copy()
print(f"Rows kept: {len(kept)} (from {len(df)})")

if len(good_prompts) < 90:
    print(f"\n⚠️  WARNING: only {len(good_prompts)} aligned prompts. "
          "RQ2 will be weak. Re-examine raw CSVs.")


# ---------------------------------------------------------------
# 4. Leakage check
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("LEAKAGE CHECK")
print("=" * 60)

kept["has_openai_img"] = kept["output_clean"].str.contains(
    "images.openai.com", case=False, regex=False)
kept["has_youtube"] = kept["output_clean"].str.contains(
    "youtube", case=False, regex=False)

print("Fraction with 'images.openai.com':")
print(kept.groupby("llm")["has_openai_img"].mean().round(3))
print("Fraction with 'youtube':")
print(kept.groupby("llm")["has_youtube"].mean().round(3))


# ---------------------------------------------------------------
# 5. Balance
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("BALANCING")
print("=" * 60)

n = kept["llm"].value_counts().min()
print(f"Balancing to {n} per class")
kept = kept.groupby("llm").sample(n=n, random_state=42).reset_index(drop=True)


# ---------------------------------------------------------------
# 6. Labels
# ---------------------------------------------------------------
le = LabelEncoder()
kept["label"] = le.fit_transform(kept["llm"])
print("Label mapping:", dict(zip(le.classes_, le.transform(le.classes_))))
pd.to_pickle(le, "models/label_encoder.pkl")


# ---------------------------------------------------------------
# 7. *** SPLIT BY PROMPT *** — key fix
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("SPLIT (grouped by prompt)")
print("=" * 60)

groups = kept["prompt_id"]

# 70/15/15 by prompt
gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
train_idx, temp_idx = next(gss1.split(kept, groups=groups))
train = kept.iloc[train_idx].reset_index(drop=True)
temp  = kept.iloc[temp_idx].reset_index(drop=True)

gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
val_idx, test_idx = next(gss2.split(temp, groups=temp["prompt_id"]))
val  = temp.iloc[val_idx].reset_index(drop=True)
test = temp.iloc[test_idx].reset_index(drop=True)

# Save (drop the helper prompt_id column)
keep_cols = ["llm", "input_clean", "output_clean", "label"]
train[keep_cols].to_csv(f"{OUT_DIR}/train.csv", index=False)
val[keep_cols].to_csv(f"{OUT_DIR}/val.csv", index=False)
test[keep_cols].to_csv(f"{OUT_DIR}/test.csv", index=False)

print(f"Train {len(train)} rows | {train['input_clean'].nunique()} prompts")
print(f"Val   {len(val)} rows | {val['input_clean'].nunique()} prompts")
print(f"Test  {len(test)} rows | {test['input_clean'].nunique()} prompts")


# ---------------------------------------------------------------
# 8. Verify NO leakage between splits
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("FINAL ALIGNMENT CHECK")
print("=" * 60)

# 8a. Each split must have all 3 LLMs per prompt
ok = True
for name, split in [("train", train), ("val", val), ("test", test)]:
    counts = split.groupby("prompt_id")["llm"].nunique().value_counts().to_dict()
    bad = sum(v for k, v in counts.items() if k != 3)
    print(f"{name:6s}: prompts-by-#LLMs = {counts}  (bad={bad})")
    if bad > 0:
        ok = False

# 8b. No prompt appears in more than one split
train_p = set(train["prompt_id"])
val_p   = set(val["prompt_id"])
test_p  = set(test["prompt_id"])
overlap_tv = train_p & val_p
overlap_tt = train_p & test_p
overlap_vt = val_p & test_p
print(f"\nPrompt overlap train∩val: {len(overlap_tv)}")
print(f"Prompt overlap train∩test: {len(overlap_tt)}")
print(f"Prompt overlap val∩test:  {len(overlap_vt)}")

# 8c. Distribution per class
print("\nTrain class distribution:")
print(train["llm"].value_counts())
print("\nVal class distribution:")
print(val["llm"].value_counts())
print("\nTest class distribution:")
print(test["llm"].value_counts())

if ok and not (overlap_tv or overlap_tt or overlap_vt):
    print("\n✅ Every split has all 3 LLMs per prompt. No prompt leakage.")
else:
    print("\n❌ Leakage detected. Investigate.")

print("\nPreprocessing complete.")