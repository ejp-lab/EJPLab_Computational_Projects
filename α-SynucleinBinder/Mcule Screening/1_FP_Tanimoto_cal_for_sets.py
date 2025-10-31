"""
For training + test datasets, calculate Tanimoto similarity to reference compounds:

    python 1_FP_Tanimoto_cal_for_sets.py \
    train_PLUS_test_data.csv \
    train_PLUS_test_fp_sim.csv

    
For prospective datasets, calculate Tanimoto similarity to reference compounds:
    python 1_FP_Tanimoto_cal_for_sets.py \
    prospective_data.csv \
    prospective_fp_sim.csv

"""

import sys, os, math, pandas as pd
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem

RDLogger.DisableLog('rdApp.*')

REFS = {
    "BV-21": "IC1=CC=C(C2=NOC(NC(C3=CC=C(OC)C=C3)=O)=C2)C=C1",
    "TZ61-84": "I/C=C/COC1=CC=C(N=C(NC2=CN=C(OC)C=C2)C=C3)C3=C1",
    "M503-1619": "COC1=CC=C(C=C1)NC(C2=C(N)N=C(N=C2)N3CCN(C4=NC=CC=C4)CC3)=O"
}
RADIUS = 3
NBITS = 1024

def smiles_to_mol(smi: str):
    if smi is None or (isinstance(smi, float) and math.isnan(smi)):
        return None
    try:
        m = Chem.MolFromSmiles(str(smi))
        if m is None:
            return None
        Chem.SanitizeMol(m)
        return m
    except Exception:
        return None

def canonicalize_smiles(smi: str):
    m = smiles_to_mol(smi)
    return Chem.MolToSmiles(m, isomericSmiles=True) if m is not None else None

def morgan_fp(m):
    if m is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(m, radius=RADIUS, nBits=NBITS)

def tanimoto(fp1, fp2):
    if fp1 is None or fp2 is None:
        return 0.0
    return DataStructs.TanimotoSimilarity(fp1, fp2)

def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    IN_PATH, OUT_PATH = sys.argv[1], sys.argv[2]

    if not os.path.exists(IN_PATH):
        raise FileNotFoundError(f"Missing input file: {IN_PATH}")

    df = pd.read_csv(IN_PATH)

    # detect smiles column
    smi_col = next((c for c in df.columns
                    if str(c).lower() in ("smiles", "smile", "smiles_canonical")), None)
    if smi_col is None:
        raise ValueError("Couldn't find a SMILES column in the input CSV.")

    # canonicalize + build mols
    df["SMILES_canonical"] = df[smi_col].astype(str).apply(canonicalize_smiles)
    mols = [smiles_to_mol(s) for s in df["SMILES_canonical"]]

    # refs (mol + fingerprints)
    ref_mols = {k: smiles_to_mol(v) for k, v in REFS.items()}
    ref_fps = {k: morgan_fp(m) for k, m in ref_mols.items()}

    rows = []
    for i, r in df.iterrows():
        m = mols[i]
        fp = morgan_fp(m)
        sims = {name: tanimoto(fp, ref_fps[name]) for name in ref_fps.keys()}
        parent = max(sims.items(), key=lambda kv: kv[1])[0] if sims else None

        row = {
            "SMILES_input": r[smi_col],
            "SMILES_canonical": r["SMILES_canonical"],
            "FP_BV-21": sims.get("BV-21", 0.0),
            "FP_TZ61-84": sims.get("TZ61-84", 0.0),
            "FP_M503-1619": sims.get("M503-1619", 0.0),
            "Parent_Label": parent,
        }
        # keep useful metadata if present
        for keep in ("Name", "label" "mol_id", "parent_label"):
            if keep in df.columns:
                row[keep] = r[keep]
        rows.append(row)

    # create a new column 'FP_Score' based on the parent label
    for row in rows:
        parent = row["Parent_Label"]
        if parent == "BV-21":
            row["FP_Score"] = row["FP_BV-21"]
        elif parent == "TZ61-84":
            row["FP_Score"] = row["FP_TZ61-84"]
        elif parent == "M503-1619":
            row["FP_Score"] = row["FP_M503-1619"]
        else:
            row["FP_Score"] = None
    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_PATH, index=False)
    print(f"Saved: {OUT_PATH}")

if __name__ == "__main__":
    main()
