'''
python 0_Tanimoto_Mcule.py \
  --smi mcule_db.smi \
  --outdir ./mcule_fp_out \
  --nprocs 64 \
  --count_lines
'''
import argparse
import os
import torch
import multiprocessing as mp
from functools import partial
from typing import List, Tuple, Dict, Optional

import pandas as pd
import numpy as np

from transformers import AutoTokenizer, AutoModel

# RDKit imports
from rdkit import Chem
from rdkit.Chem import rdFMCS
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors as rdmd
from rdkit import RDLogger
from rdkit import DataStructs

# Progress
from tqdm import tqdm
import pdb

# Silence RDKit warnings
RDLogger.DisableLog('rdApp.*')

# Try visualization backends
from matplotlib import pyplot as plt

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

REFS = {
    "BV-21": "IC1=CC=C(C2=NOC(NC(C3=CC=C(OC)C=C3)=O)=C2)C=C1",
    "TZ61-84": "I/C=C/COC1=CC=C(N=C(NC2=CN=C(OC)C=C2)C=C3)C3=C1",
    "M503-1619": "COC1=CC=C(C=C1)NC(C2=C(N)N=C(N=C2)N3CCN(C4=NC=CC=C4)CC3)=O"
}
PARENT = ["M503-1619", "BV-21", "TZ61-84"]
MIN_SIM_THRESHOLD = 0.1


def compute_fp(smiles: str, radius: int, nBits: int):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)
    except Exception:
        return None

def jaccard_fp(fp1, fp2) -> float:
    if fp1 is None or fp2 is None:
        return 0.0
    # For bit vectors, Jaccard == Tanimoto
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def _best_parent(sim_map: Dict[str, float]) -> str:
        if not sim_map:
            return "OTHER"
        max_sim = max(sim_map.values())
        if max_sim <= MIN_SIM_THRESHOLD:
            return "OTHER"
        # Stable tie-break using PARENT_ORDER
        best = None
        for parent in PARENT:
            if sim_map.get(parent, -1.0) == max_sim:
                best = parent
                break
        return best if best is not None else max(sim_map.items(), key=lambda kv: kv[1])[0]
    

def init_refs():
    ref_fps = {}
    for k, v in REFS.items():
        ref_fps[k] = compute_fp(v, radius=3, nBits=1024)
    return ref_fps

# Worker—initialized globals to reduce pickling overhead
_REF_FPS = None

def _init_worker():
    global _REF_FPS
    _REF_FPS = init_refs()

def score_one_fp(record: Tuple[str, Optional[str]], radius: int = 3, nBits: int = 1024):
    "record = (smiles, mol_id) from the .smi file."
    global _REF_FPS
    raw_smi, mol_id = record
    fp = compute_fp(raw_smi, radius=radius, nBits=nBits)
    sims = {name: jaccard_fp(fp, rfp) for name, rfp in _REF_FPS.items()} if _REF_FPS else {}

    best_label = _best_parent(sims)
    return {
        "mol_id": mol_id,
        "smiles": raw_smi,
        # Keep the same output column names, now meaning FP-Jaccard/Tanimoto
        "bv_21_jacc": sims.get("BV-21", 0.0),
        "tz61_84_jacc": sims.get("TZ61-84", 0.0),
        "m503_1619_jacc": sims.get("M503-1619", 0.0),
        "parent_label": best_label,
    }

def stream_smi(path: str, smiles_col: int = 0, id_col: Optional[int] = 1):
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            smi = parts[smiles_col] if len(parts) > smiles_col else None
            mol_id = parts[id_col] if id_col is not None and len(parts) > id_col else None
            yield (smi, mol_id)

def count_lines(path: str) -> int:
    n = 0
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for _ in f:
            n += 1
    return n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smi", required=True, help="Path to MCULE .smi file (SMILES [ID])")
    ap.add_argument("--outdir", default="./mcule_fp_out", help="Output directory")
    ap.add_argument("--nprocs", type=int, default=64, help="Worker processes")
    ap.add_argument("--chunk", type=int, default=25000, help="Records per batch (memory vs throughput)")
    ap.add_argument("--limit", type=int, default=0, help="Process only first N records (0 = all)")
    ap.add_argument("--total_hint", type=int, default=0, help="Optional total rows hint for progress (if not counting)")
    ap.add_argument("--count_lines", action="store_true", help="Scan file to count total rows for accurate ETA")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    embed_out = os.path.join(args.outdir, "mcule_fp_label.parquet")

    # Determine total for progress bar
    total_rows = None
    if args.count_lines:
        total_rows = count_lines(args.smi)
    elif args.total_hint > 0:
        total_rows = args.total_hint
    if args.limit and args.limit > 0:
        total_rows = min(total_rows, args.limit) if total_rows is not None else args.limit

    # Prepare multiprocessing
    ctx = mp.get_context("spawn")  # safer with heavy libs
    pool = ctx.Pool(processes=args.nprocs, initializer=_init_worker, maxtasksperchild=500)
    worker = partial(score_one_fp)

    # label + metrics (with progress)
    results = []
    total = 0
    batch = []

    proc_bar = tqdm(total=total_rows, desc="Scoring FP Jaccard", unit="mol")

    def flush_batch(b):
        nonlocal results
        if not b:
            return
        for row in pool.imap_unordered(worker, b, chunksize=32):
            results.append(row)
            proc_bar.update(1)

    for rec in stream_smi(args.smi):
        batch.append(rec)
        total += 1
        if args.limit and total > args.limit:
            break
        if len(batch) >= args.chunk:
            flush_batch(batch)
            batch = []
    if batch:
        flush_batch(batch)

    pool.close()
    pool.join()
    proc_bar.close()

    df = pd.DataFrame(results)
    df.to_parquet(embed_out, index=False)

if __name__ == "__main__":
    main()
