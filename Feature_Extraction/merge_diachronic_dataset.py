# =======================================
# merge_diachronic_dataset.py (v3 - drop NaN labels)
# =======================================
import pandas as pd
from config import TIMEDIARIES_PARQUET, OUTPUT_DIR
from utils import read_parquet_folder, save_csv

def merge_behavior_gt():
    print("Loading behavioral (v2) and GT data...")

    # Load sensors (already has no NaNs in feature columns)
    sensors = pd.read_csv(f"{OUTPUT_DIR}/sensor_features_halfhour_v2.csv")
    sensors["bin"] = pd.to_datetime(sensors["bin"])

    # Load GT (diary)
    diary = read_parquet_folder(TIMEDIARIES_PARQUET)
    diary["instancetimestamp"] = pd.to_datetime(
        diary["instancetimestamp"], format="%d-%m-%Y %H:%M", errors="coerce"
    )
    diary = diary.dropna(subset=["instancetimestamp"])   # drop invalid timestamps
    diary["bin"] = diary["instancetimestamp"].dt.floor("30min")

    # keep only needed columns
    gt_cols = ["userid", "bin", "A6a"]
    diary = diary[gt_cols]

    # Merge (inner keeps only bins having both sensor + GT)
    df = sensors.merge(diary, on=["userid", "bin"], how="inner")

    #  Important change: DROP rows where the GT label is missing
    df = df.dropna(subset=["A6a"])

    # Keep feature NaNs as 0 if any (rare)
    feature_cols = df.columns.difference(["A6a"])
    df[feature_cols] = df[feature_cols].fillna(0)

    save_csv(df, f"{OUTPUT_DIR}/final_dataset_halfhour_v2.csv")
    print(f"Final dataset shape: {df.shape}")
    print("NaN GT labels dropped safely!")

if __name__ == "__main__":
    merge_behavior_gt()
