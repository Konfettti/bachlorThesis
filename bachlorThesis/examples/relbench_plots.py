# relbench_plots.py
# Vergleiche Modelle/Depth/Aggregates anhand deiner geloggten Ergebnisse.

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================
# Konfiguration
# ==========================
SAVE_FIGS = True            # auf False setzen, wenn du keine PNGs speichern willst
OUTDIR = Path("relbench_figs")
OUTDIR.mkdir(exist_ok=True)

# ==========================
# Daten aus deinem Log
# ==========================
records = []

def add(model, depth, agg, split, r2, mae, rmse):
    records.append({
        "model": model,
        "depth": depth,
        "agg_features": "with-agg" if agg else "no-agg",
        "split": split,
        "R2": float(r2),
        "MAE": float(mae),
        "RMSE": float(rmse),
    })

# -------------------- Block 1: KEINE Aggregates (Flag gesetzt, aber leer) --------------------
# depth = 1
add("tabpfn", 1, False, "train", 0.0024, 0.6960, 0.9837)
add("tabpfn", 1, False, "val",  -0.2281, 0.5679, 0.6306)
add("tabpfn", 1, False, "test", -0.1262, 0.6509, 0.8382)

add("xgboost", 1, False, "train", 0.3076, 0.5833, 0.8195)
add("xgboost", 1, False, "val",  -0.2686, 0.5342, 0.6409)
add("xgboost", 1, False, "test", -0.1676, 0.6066, 0.8534)

add("lightgbm", 1, False, "train", 0.0140, 0.6760, 0.9779)
add("lightgbm", 1, False, "val",  -0.1526, 0.5410, 0.6109)
add("lightgbm", 1, False, "test", -0.0879, 0.6170, 0.8238)

# depth = 2
add("tabpfn", 2, False, "train", 0.0076, 0.6947, 0.9811)
add("tabpfn", 2, False, "val",  -0.2454, 0.5726, 0.6350)
add("tabpfn", 2, False, "test", -0.1250, 0.6515, 0.8377)

add("xgboost", 2, False, "train", 0.2412, 0.6067, 0.8579)
add("xgboost", 2, False, "val",  -0.2662, 0.5391, 0.6403)
add("xgboost", 2, False, "test", -0.1360, 0.5972, 0.8418)

add("lightgbm", 2, False, "train", 0.0140, 0.6760, 0.9779)
add("lightgbm", 2, False, "val",  -0.1526, 0.5410, 0.6109)
add("lightgbm", 2, False, "test", -0.0879, 0.6170, 0.8238)


# -------------------- Block 2: MIT Aggregates (["count","num_unique","mode","percent_true","max","min","std"]) --------------------
# depth = 1
add("tabpfn", 1, True, "train", 0.0024, 0.6960, 0.9837)
add("tabpfn", 1, True, "val",  -0.2281, 0.5679, 0.6306)
add("tabpfn", 1, True, "test", -0.1262, 0.6509, 0.8382)

add("xgboost", 1, True, "train", 0.3076, 0.5833, 0.8195)
add("xgboost", 1, True, "val",  -0.2686, 0.5342, 0.6409)
add("xgboost", 1, True, "test", -0.1676, 0.6066, 0.8534)

add("lightgbm", 1, True, "train", 0.0140, 0.6760, 0.9779)
add("lightgbm", 1, True, "val",  -0.1526, 0.5410, 0.6109)
add("lightgbm", 1, True, "test", -0.0879, 0.6170, 0.8238)

# depth = 2
add("tabpfn", 2, True, "train", 0.0054, 0.6937, 0.9822)
add("tabpfn", 2, True, "val",  -0.2261, 0.5676, 0.6301)
add("tabpfn", 2, True, "test", -0.1223, 0.6482, 0.8367)

add("xgboost", 2, True, "train", 0.2412, 0.6067, 0.8579)
add("xgboost", 2, True, "val",  -0.2662, 0.5391, 0.6403)
add("xgboost", 2, True, "test", -0.1360, 0.5972, 0.8418)

add("lightgbm", 2, True, "train", 0.0140, 0.6760, 0.9779)
add("lightgbm", 2, True, "val",  -0.1526, 0.5410, 0.6109)
add("lightgbm", 2, True, "test", -0.0879, 0.6170, 0.8238)

df = pd.DataFrame.from_records(records)

# Ordnung/Labels
depth_order = [1, 2, 10]
agg_order = ["no-agg", "with-agg"]
df["depth"] = pd.Categorical(df["depth"], categories=depth_order, ordered=True)
df["agg_features"] = pd.Categorical(df["agg_features"], categories=agg_order, ordered=True)

# ==========================
# Tabelle optional speichern
# ==========================
csv_path = OUTDIR / "relbench_results_compiled.csv"
df.to_csv(csv_path, index=False)
print(f"CSV gespeichert: {csv_path.resolve()}")

# ==========================
# TEST-only für faire Vergleiche
# ==========================
test_df = df[df["split"] == "test"].copy()
test_df["setting"] = test_df["depth"].astype(str) + " / " + test_df["agg_features"].astype(str)

# gewünschte Reihenfolge der Settings in den Plots
desired_settings = [f"{d} / no-agg" for d in depth_order] + [f"{d} / with-agg" for d in [1, 2]]
test_df["setting"] = pd.Categorical(test_df["setting"], categories=desired_settings, ordered=True)

# ==========================
# Hilfsfunktionen für Plots
# ==========================
def bar_by_setting(metric):
    pivot = test_df.pivot_table(index="setting", columns="model", values=metric, aggfunc="mean")
    ax = pivot.plot(kind="bar", figsize=(10, 5))
    ax.set_title(f"{metric} auf TEST nach Setting (depth / aggregates)")
    ax.set_ylabel(metric)
    ax.set_xlabel("Setting (depth / aggregates)")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    if SAVE_FIGS:
        p = OUTDIR / f"bar_{metric}_by_setting.png"
        plt.savefig(p, dpi=150)
        print(f"Fig gespeichert: {p.resolve()}")
    plt.show()

def line_per_model(model):
    sub = test_df[test_df["model"] == model].copy()
    plt.figure(figsize=(8, 4))
    for agg in agg_order:
        sub2 = sub[sub["agg_features"] == agg].sort_values("depth")
        if not sub2.empty:
            plt.plot(sub2["depth"].astype(int), sub2["R2"], marker="o", label=agg)
    plt.title(f"TEST R² vs depth — {model}")
    plt.xlabel("depth")
    plt.ylabel("R²")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    if SAVE_FIGS:
        p = OUTDIR / f"line_R2_vs_depth_{model}.png"
        plt.savefig(p, dpi=150)
        print(f"Fig gespeichert: {p.resolve()}")
    plt.show()

# ==========================
# Plots erzeugen
# ==========================
for metric in ["R2", "MAE", "RMSE"]:
    bar_by_setting(metric)

for m in sorted(test_df["model"].unique()):
    line_per_model(m)

# ==========================
# Kompakte Text-Zusammenfassung
# ==========================
summary_lines = []
for model in sorted(test_df["model"].unique()):
    def getR2(setting):
        row = test_df[(test_df["model"]==model) & (test_df["setting"]==setting)]
        return float(row["R2"]) if not row.empty else float("nan")
    txt = (
        f"{model}: "
        f"depth1_no={getR2('1 / no-agg'):.4f}, "
        f"depth2_no={getR2('2 / no-agg'):.4f}, "
        f"depth10_no={getR2('10 / no-agg'):.4f}"
    )
    d1a, d2a = getR2('1 / with-agg'), getR2('2 / with-agg')
    add_parts = []
    if not pd.isna(d1a): add_parts.append(f"depth1_agg={d1a:.4f}")
    if not pd.isna(d2a): add_parts.append(f"depth2_agg={d2a:.4f}")
    if add_parts:
        txt += " | " + ", ".join(add_parts)
    summary_lines.append(txt)

print("\n".join(summary_lines))
