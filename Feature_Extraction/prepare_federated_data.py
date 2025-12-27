import pandas as pd, os
from config import OUTPUT_DIR

def prepare_federated_data():
    df = pd.read_csv(f"{OUTPUT_DIR}/final_dataset_halfhour_v2.csv")
    user_groups = dict(tuple(df.groupby("userid")))
    fed_dir = os.path.join(OUTPUT_DIR, "federated_users_halfhour")
    os.makedirs(fed_dir, exist_ok=True)
    for uid, d in user_groups.items():
        d.to_csv(os.path.join(fed_dir, f"user_{uid}.csv"), index=False)
    print(f"✅ Created {len(user_groups)} user datasets")

if __name__ == "__main__":
    prepare_federated_data()
