# =======================================
# utils.py
# =======================================
import pandas as pd
import pyarrow.dataset as ds
import os
import numpy as np

def read_parquet_folder(folder_path, columns=None):
    dataset = ds.dataset(folder_path, format="parquet")
    df = dataset.to_table(columns=columns).to_pandas()
    return df

def save_csv(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"✅ Saved: {path}")

# --- Helper to compute entropy safely ---
def compute_entropy(series):
    counts = series.value_counts()
    probs = counts / counts.sum()
    return -(probs * np.log2(probs + 1e-9)).sum()
