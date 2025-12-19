# relbench_user_repeat_plots.py
# Vergleiche Modelle/Depth/Aggregates anhand deiner user-repeat Klassifikations-Ergebnisse.

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================
# Konfiguration
# ==========================
SAVE_FIGS = True
OUTDIR = Path("relbench_user_repeat_figs")
OUTDIR.mkdir(exist_ok=True)

records = []

def add(model, depth, agg, split, acc, f1):
    records.append({
        "model": model,
        "depth": depth,
        "agg_features": "with-agg" if agg else "no-agg",
        "split": split,
        "Accuracy": float(acc),
        "F1": float(f1),
    })

# ==========================
# Block 1: Keine Aggregates
# ==========================
# depth = 1
add("tabpfn", 1, False, "train", 0.7550, 0.7540)
add("tabpfn", 1, False, "val", 0.5200, 0.5152)
add("tabpfn", 1, False, "test", 0.5050, 0.4995)

add("xgboost", 1, False, "train", 0.7800, 0.7789)
add("xgboost", 1, False, "val", 0.5550, 0.5500)
add("xgboost", 1, False, "test", 0.5200, 0.5141)

add("lightgbm", 1, False, "train", 0.6450, 0.6059)
add("lightgbm", 1, False, "val", 0.5400, 0.4758)
add("lightgbm", 1, False, "test", 0.5750, 0.5137)

# depth = 2
add("tabpfn", 2, False, "train", 0.7350, 0.7331)
add("tabpfn", 2, False, "val", 0.4900, 0.4825)
add("tabpfn", 2, False, "test", 0.4950, 0.4870)

add("xgboost", 2, False, "train", 0.7800, 0.7789)
add("xgboost", 2, False, "val", 0.5550, 0.5500)
add("xgboost", 2, False, "test", 0.5200, 0.5141)

add("lightgbm", 2, False, "train", 0.6450, 0.6059)
add("lightgbm", 2, False, "val", 0.5400, 0.4758)
add("lightgbm", 2, False, "test", 0.5750, 0.5137)

# ==========================
# Block 2: Mit Aggregates
# ==========================
# depth = 1
add("tabpfn", 1, True, "train", 0.7550, 0.7540)
add("tabpfn", 1, True, "val", 0.5200, 0.5152)
add("tabpfn", 1, True, "test", 0.5050, 0.4995)

add("xgboost", 1, True, "train", 0.7800, 0.7789)
add("xgboost", 1, True, "val", 0.5550, 0.5500)
add("xgboost", 1, True, "test", 0.5200, 0.5141)

add("lightgbm", 1, True, "train", 0.6450, 0.6059)
add("lightgbm", 1, True, "val", 0.5400, 0.4758)
add("lightgbm", 1, True, "test", 0.5750, 0.5137)

# depth = 2
add("tabpfn", 2, True, "train", 0.7950, 0.7931)
add("tabpfn", 2, True, "val", 0.5300, 0.5219)
add("tabpfn", 2, True, "test", 0.5250, 0.5162)

add("xgboost", 2, True, "train", 0.7800, 0.7789)
add("xgboost", 2, True, "val", 0.5550, 0.5500)
add("xgboost", 2, True, "test", 0.5200, 0.5141)

add("lightgbm", 2, True, "train", 0.6450, 0.6059)
add("lightgbm", 2, True, "val", 0.5400, 0.4758)
add("lightgbm", 2, True, "test", 0.5750, 0.5137)

# ==========================
# DataFrame + Settings
# ==========================
df = pd.DataFrame.from_records(records)
depth_order = [1, 2]
agg_order = ["no-agg", "with-agg"]
df["depth"] = pd.Categorical(df["depth"], categories=depth_order, ordered=True)
df["agg_features"] = pd.Categorical(df["agg_features"], categories=agg_order, ordered=True)
df["setting"] = df["depth"].astype(str) + " / " + df["agg_features"].astype(str)

# Nur TEST-Split für Vergleiche
test_df = df[df["split"] == "test"].copy()
desired_settings = [f"{d} / no-agg" for d in depth_order] + [f"{d} / with-agg" for d in depth_order]
test_df["setting"] = pd.Categorical(test_df["setting"], categories=desired_settings, ordered=True)

# ==========================
# Balkenplots (Accuracy, F1)
# ==========================
def bar_by_setting(metric):
    pivot = test_df.pivot_table(index="setting", columns="model", values=metric, aggfunc="mean")
    ax = pivot.plot(kind="bar", figsize=(9, 5))
    ax.set_title(f"{metric} (TEST) nach depth / aggregates")
    ax.set_ylabel(metric)
    ax.set_xlabel("Setting (depth / aggregates)")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    if SAVE_FIGS:
        p = OUTDIR / f"bar_{metric}_by_setting.png"
        plt.savefig(p, dpi=150)
        print(f"Gespeichert: {p.resolve()}")
    plt.show()

for metric in ["Accuracy", "F1"]:
    bar_by_setting(metric)

# ==========================
# Linienplots je Modell
# ==========================
def line_per_model(model, metric):
    sub = test_df[test_df["model"] == model]
    plt.figure(figsize=(7,4))
    for agg in agg_order:
        sub2 = sub[sub["agg_features"] == agg].sort_values("depth")
        if not sub2.empty:
            plt.plot(sub2["depth"].astype(int), sub2[metric], marker="o", label=agg)
    plt.title(f"{metric} vs depth — {model}")
    plt.xlabel("depth")
    plt.ylabel(metric)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    if SAVE_FIGS:
        p = OUTDIR / f"line_{metric}_vs_depth_{model}.png"
        plt.savefig(p, dpi=150)
        print(f"Gespeichert: {p.resolve()}")
    plt.show()

for metric in ["Accuracy", "F1"]:
    for m in sorted(test_df["model"].unique()):
        line_per_model(m, metric)

# ==========================
# Übersichtliche Textausgabe
# ==========================
for model in sorted(test_df["model"].unique()):
    sub = test_df[test_df["model"] == model]
    print(f"\n{model}:")
    for s in desired_settings:
        r = sub[sub["setting"] == s]
        if not r.empty:
            print(f"  {s}: Accuracy={float(r['Accuracy']):.3f}, F1={float(r['F1']):.3f}")
