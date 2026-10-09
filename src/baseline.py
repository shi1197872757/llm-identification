"""
Baselines for RQ1 and RQ2. Report-ready numbers, no convergence warnings.

Conditions:
  - RAW      : original text
  - STRIPPED : markdown/LaTeX markers removed (**, ##, $, |, `, ```)
"""
import re
import warnings
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report

warnings.filterwarnings("ignore", category=UserWarning)   # silence convergence
RANDOM_STATE = 42


# ---------------------------------------------------------------
# Load
# ---------------------------------------------------------------
tr = pd.read_csv("data/processed/train.csv")
va = pd.read_csv("data/processed/val.csv")
te = pd.read_csv("data/processed/test.csv")
print(f"Train {len(tr)} | Val {len(va)} | Test {len(te)}")


# ---------------------------------------------------------------
# Markdown stripper
# ---------------------------------------------------------------
MARKDOWN_RE = re.compile(
    r"```.*?```"                                # code fences + content
    r"|`[^`]+`"                                 # inline code
    r"|\*\*[^*]+\*\*"                           # bold
    r"|__[^_]+__"                               # bold alt
    r"|\*[^*]+\*"                               # italic
    r"|^#{1,6}\s+"                              # headers
    r"|!\[[^\]]*\]\([^)]*\)"                    # images
    r"|\[[^\]]*\]\([^)]*\)"                     # links
    r"|^\s*\|.*\|\s*$"                          # table rows
    r"|^\s*[-*+]\s+"                            # bullets
    r"|\${1,2}[^$]+\${1,2}",                    # inline/display latex
    flags=re.MULTILINE | re.DOTALL,
)


def strip_markdown(text):
    if not isinstance(text, str):
        return ""
    text = MARKDOWN_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------
# Content-loss report
# ---------------------------------------------------------------
print("\n--- Content loss after stripping markdown ---")
for llm, g in tr.groupby("llm"):
    raw_chars = g["output_clean"].str.len().sum()
    stripped_chars = g["output_clean"].map(strip_markdown).str.len().sum()
    pct = stripped_chars / max(raw_chars, 1) * 100
    print(f"  {llm:9s} raw={raw_chars:7d}  stripped={stripped_chars:7d}  ({pct:.0f}% kept)")


# ---------------------------------------------------------------
# Feature extractor
# ---------------------------------------------------------------
def format_features(df, text_col="output_clean"):
    s = df[text_col].fillna("").astype(str)
    return pd.DataFrame({
        "n_chars":     s.str.len(),
        "n_words":     s.str.split().str.len(),
        "n_newlines":  s.str.count("\n"),
        "n_commas":    s.str.count(","),
        "n_hashes":    s.str.count("#"),
        "n_dollars":   s.str.count(r"\$"),
        "n_asterisks": s.str.count(r"\*"),
        "n_pipes":     s.str.count(r"\|"),
        "n_backticks": s.str.count("`"),
    })


def evaluate(name, y_true, y_pred, print_report=False):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    print(f"\n--- {name} ---")
    print(f"  acc      : {acc:.3f}")
    print(f"  macro_f1 : {f1:.3f}")
    if print_report:
        print(classification_report(y_true, y_pred, digits=3, zero_division=0))
    return acc, f1


# ---------------------------------------------------------------
# Baseline 1 — Length + format (raw)
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("BASELINE 1 — Length + format features (RAW)")
print("=" * 60)

Xtr = format_features(tr)
Xte = format_features(te)
clf = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
)
clf.fit(Xtr, tr["label"])
acc_fmt_raw, f1_fmt_raw = evaluate(
    "Length+format (raw)", te["label"], clf.predict(Xte))

# Feature importance from the logistic regression step
coefs = pd.Series(
    abs(clf.named_steps["logisticregression"].coef_).mean(axis=0),
    index=Xtr.columns,
).sort_values(ascending=False)
print("\n  Feature importance (mean |coef|, standardized):")
for k, v in coefs.items():
    print(f"    {k:12s} {v:.3f}")


# ---------------------------------------------------------------
# Baseline 2 — Length + format (stripped)
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("BASELINE 2 — Length + format features (STRIPPED)")
print("=" * 60)

tr["output_stripped"] = tr["output_clean"].map(strip_markdown)
te["output_stripped"] = te["output_clean"].map(strip_markdown)

Xtr = format_features(tr, "output_stripped")
Xte = format_features(te, "output_stripped")
clf = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
)
clf.fit(Xtr, tr["label"])
acc_fmt_str, f1_fmt_str = evaluate(
    "Length+format (stripped)", te["label"], clf.predict(Xte))
print(f"\n  Δ (raw − stripped): acc {acc_fmt_raw - acc_fmt_str:+.3f}")


# ---------------------------------------------------------------
# Baseline 3 — TF-IDF on INPUT only
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("BASELINE 3 — TF-IDF on INPUT only (RQ2 control)")
print("=" * 60)

vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
Xtr = vec.fit_transform(tr["input_clean"].astype(str))
Xte = vec.transform(te["input_clean"].astype(str))
clf = LogisticRegression(max_iter=5000, random_state=RANDOM_STATE).fit(Xtr, tr["label"])
acc_in, f1_in = evaluate("TF-IDF on INPUT", te["label"], clf.predict(Xte),
                         print_report=True)


# ---------------------------------------------------------------
# Baseline 4 — TF-IDF on OUTPUT (raw)
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("BASELINE 4 — TF-IDF on OUTPUT (RAW, RQ1)")
print("=" * 60)

vec = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2)
Xtr = vec.fit_transform(tr["output_clean"].astype(str))
Xte = vec.transform(te["output_clean"].astype(str))
clf = LogisticRegression(max_iter=5000, random_state=RANDOM_STATE).fit(Xtr, tr["label"])
acc_out_raw, f1_out_raw = evaluate("TF-IDF on OUTPUT (raw)", te["label"],
                                    clf.predict(Xte), print_report=True)


# ---------------------------------------------------------------
# Baseline 5 — TF-IDF on OUTPUT (stripped)
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("BASELINE 5 — TF-IDF on OUTPUT (STRIPPED)")
print("=" * 60)

vec = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2)
Xtr = vec.fit_transform(tr["output_stripped"].astype(str))
Xte = vec.transform(te["output_stripped"].astype(str))
clf = LogisticRegression(max_iter=5000, random_state=RANDOM_STATE).fit(Xtr, tr["label"])
acc_out_str, f1_out_str = evaluate("TF-IDF on OUTPUT (stripped)", te["label"],
                                    clf.predict(Xte), print_report=True)
print(f"\n  Δ (raw − stripped): acc {acc_out_raw - acc_out_str:+.3f}")


# ---------------------------------------------------------------
# Baseline 6 — TF-IDF on INPUT + OUTPUT (raw)
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("BASELINE 6 — TF-IDF on INPUT + OUTPUT (RAW)")
print("=" * 60)

vec = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2)
Xtr = vec.fit_transform(
    tr["input_clean"].astype(str) + " [SEP] " + tr["output_clean"].astype(str))
Xte = vec.transform(
    te["input_clean"].astype(str) + " [SEP] " + te["output_clean"].astype(str))
clf = LogisticRegression(max_iter=5000, random_state=RANDOM_STATE).fit(Xtr, tr["label"])
acc_both, f1_both = evaluate("TF-IDF on INPUT+OUTPUT", te["label"], clf.predict(Xte))


# ---------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("BASELINE SUMMARY (test set, n=45)")
print("=" * 60)

def row(name, acc, f1):
    print(f"{name:<34s} {acc:7.3f} {f1:9.3f}")

print(f"{'Condition':<34s} {'Acc':>7s} {'MacroF1':>9s}")
print("-" * 52)
print(f"{'Chance (3 classes)':<34s} {1/3:7.3f} {1/3:9.3f}")
print("-" * 52)
row("Length+format (raw)",         acc_fmt_raw, f1_fmt_raw)
row("Length+format (stripped)",    acc_fmt_str, f1_fmt_str)
print("-" * 52)
row("TF-IDF on INPUT only (RQ2)",  acc_in,      f1_in)
row("TF-IDF on OUTPUT (raw)",      acc_out_raw, f1_out_raw)
row("TF-IDF on OUTPUT (stripped)", acc_out_str, f1_out_str)
row("TF-IDF on INPUT+OUTPUT",      acc_both,    f1_both)
print("-" * 52)

print(f"\nKey deltas:")
print(f"  Formatting signal (raw - stripped, length feats): "
      f"{acc_fmt_raw - acc_fmt_str:+.3f}")
print(f"  Content signal    (raw - stripped, TF-IDF output): "
      f"{acc_out_raw - acc_out_str:+.3f}")
print(f"  Prompt adds       (both - output, raw): "
      f"{acc_both - acc_out_raw:+.3f}")
print(f"  Input alone vs chance: {acc_in - 1/3:+.3f}")

print("\nInterpretation:")
if abs(acc_in - 1/3) < 0.05:
    print("  ✅ RQ2 control is CLEAN: input-only ≈ chance.")
else:
    print(f"  ⚠️  RQ2 control is NOT clean: input-only = {acc_in:.3f}.")

if acc_fmt_raw - acc_fmt_str > 0.15:
    print("  📌 Formatting markers carry substantial signal "
          "(RQ4-style finding).")

if acc_out_str > 1/3 + 0.15:
    print("  📌 Even after stripping markdown, TF-IDF on output beats chance. "
          "Content/word choice carries real signal.")

if acc_both <= acc_out_raw:
    print("  📌 Adding the prompt does NOT improve classification "
          "(supports RQ2 null hypothesis).")