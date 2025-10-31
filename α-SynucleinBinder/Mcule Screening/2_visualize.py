
"""
python 2_visualize.py train_PLUS_test_fp_sim.csv prospective_fp_sim.csv mcule_fp_label.parquet

"""

import sys
import matplotlib.pyplot as plt
import pyarrow.dataset as ds
import pyarrow.compute as pc
import numpy as np
import pandas as pd


# ----------------------------
# Command-line arguments
# ----------------------------
if len(sys.argv) != 4:
    print(__doc__)
    sys.exit(1)

TRAIN_TEST_CSV = sys.argv[1]
PROSPECTIVE_CSV = sys.argv[2]
PARQUET_PATH = sys.argv[3]

# ----------------------------
# Load input files
# ----------------------------
train_test = pd.read_csv(TRAIN_TEST_CSV)
prospective = pd.read_csv(PROSPECTIVE_CSV)
dataset = ds.dataset(PARQUET_PATH, format="parquet")

# ----------------------------
# Parent mapping: column name ↔ display label
# ----------------------------
parent_to_info = {
    "BV-21": ("bv_21_jacc", "BV-21"),
    "TZ61-84": ("tz61_84_jacc", "TZ61-84"),
    "M503-1619": ("m503_1619_jacc", "M503-1619"),
}

# ----------------------------
# Visualization setup
# ----------------------------
fig, axes = plt.subplots(
    nrows=1, ncols=3, figsize=(12, 4), sharey=True, constrained_layout=True
)
bins = np.linspace(0, 1, 50)

for ax, (parent_key, (column, display_name)) in zip(axes, parent_to_info.items()):
    # Load large parquet column by streaming batches
    vals = []
    for batch in dataset.to_batches(columns=[column], batch_size=1_000_000):
        arr = batch[column]
        vals.append(pc.drop_null(arr).to_numpy(zero_copy_only=False))
    all_vals = np.concatenate(vals) if vals else np.array([])

    # Base histogram (gray background)
    ax.hist(all_vals, bins=bins, color="lightgray", edgecolor="none", alpha=0.7)

    # Determine overlay heights
    y_max = ax.get_ylim()[1]
    h_red = y_max * 0.05
    h_blue = y_max * 0.07

    # Overlay Train+Test (red) and Prospective (blue) vlines
    tt_scores = train_test.loc[
        train_test["Parent_Label"] == parent_key, "FP_Score"
    ].dropna().to_numpy()
    pr_scores = prospective.loc[
        prospective["Parent_Label"] == parent_key, "FP_Score"
    ].dropna().to_numpy()

    if tt_scores.size:
        ax.vlines(tt_scores, 0, h_red, color="red", linewidth=1, label="Train+Test")
    if pr_scores.size:
        ax.vlines(pr_scores, 0, h_blue, color="blue", linewidth=1, label="Prospective")

    # Styling
    ax.set_title(display_name, fontsize=14, fontweight="bold", family="sans-serif", pad=8)
    ax.set_xlabel("FP Jaccard", fontsize=12, fontstyle="italic", family="serif", fontweight="semibold")
    ax.tick_params(axis="both", labelsize=10)
    ax.set_xlim(0, 1)
    ax.legend(loc="upper right", fontsize=9, frameon=False)

axes[0].set_ylabel("Frequency", fontsize=12, fontstyle="italic", family="serif", fontweight="semibold")

# Save and show
plt.savefig("train_test_vs_prospective_fp_sim_overlay.png", dpi=400, bbox_inches="tight")
plt.show()

print("Saved figure: train_test_vs_prospective_fp_sim_overlay.png")
