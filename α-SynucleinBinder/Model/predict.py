
"""
python predict.py \
    cv_selected_logistic_regression.joblib \
    cv_selected_features.joblib \
    prospective_mordred.csv \
    prospective_data.csv \
    prospective_data_predictions.csv \
    SMILES
"""

import sys
import numpy as np
import pandas as pd
import joblib
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler


# ----------------------------- #
#        Helper function        #
# ----------------------------- #
def morgan_bits_1024_radius3(smiles):
    """Generate 1024-bit Morgan fingerprint (radius 3)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=1024)
    arr = np.zeros((1024,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


# ----------------------------- #
#           Main code           #
# ----------------------------- #
if len(sys.argv) != 7:
    print(__doc__)
    sys.exit(1)

MODEL_PATH, FEATS_PATH, MORDRED_CSV, RAW_CSV, OUT_PRED, SMILES_col = sys.argv[1:7]

# Load model and features
model = joblib.load(MODEL_PATH)
selected_feats = joblib.load(FEATS_PATH)
if isinstance(selected_feats, np.ndarray):
    selected_feats = selected_feats.tolist()

# Load data
df_mord = pd.read_csv(MORDRED_CSV)
df_raw = pd.read_csv(RAW_CSV)

df_eval = df_mord.merge(df_raw[[SMILES_col, "label"]], how="inner",
                        on=SMILES_col, suffixes=("", "_raw"))

# Determine which bits to include
needed_bit_idxs = sorted(
    int(name.split("_", 1)[1])
    for name in selected_feats
    if name.startswith("Bit_")
)

# Build Morgan bits
need_bits = len(needed_bit_idxs) > 0
bit_mat = np.zeros((len(df_eval), len(needed_bit_idxs)), dtype=np.int8) if need_bits else None

failed_idx = []
if need_bits:
    for i, smi in enumerate(df_eval[SMILES_col].astype(str).tolist()):
        arr = morgan_bits_1024_radius3(smi)
        if arr is None:
            failed_idx.append(i)
            continue
        bit_mat[i, :] = arr[needed_bit_idxs]

df_bits = pd.DataFrame(bit_mat, columns=[f"Bit_{j}" for j in needed_bit_idxs], index=df_eval.index) if need_bits else pd.DataFrame(index=df_eval.index)

# Handle Mordred features
mordred_needed = [name for name in selected_feats if not name.startswith("Bit_")]
for c in mordred_needed:
    if c not in df_eval.columns:
        df_eval[c] = 0.0

# Build feature matrix
parts = []
for name in selected_feats:
    if name.startswith("Bit_"):
        parts.append(df_bits[[name]].values if name in df_bits.columns else np.zeros((len(df_eval), 1), dtype=np.float32))
    else:
        parts.append(df_eval[[name]].values.astype(np.float32))
X = np.hstack(parts).astype(np.float32)

# Scale features
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Predict
probs = model.predict_proba(X)[:, 1]
pred_label = (probs >= 0.5).astype(int)
y_true = df_eval["label"].astype(int).values

# Report
report = classification_report(y_true, pred_label, digits=2)
print("=== Classification report ===")
print(report)
if failed_idx:
    print(f"NOTE: {len(failed_idx)} SMILES failed to parse; bits set to 0.")

# Save predictions
pred_df = pd.DataFrame({
    SMILES_col: df_eval[SMILES_col].values,
    "pred_prob": probs,
    "pred_label": pred_label,
    "label_true": y_true
})
pred_df.to_csv(OUT_PRED, index=False)
print(f"Predictions saved to: {OUT_PRED}")
