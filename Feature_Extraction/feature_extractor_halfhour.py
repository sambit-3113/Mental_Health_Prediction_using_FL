# =======================================
# feature_extractor_halfhour_v2.py  (streaming / low-memory, final - walking & running removed)
# =======================================
"""
Streaming feature extractor that reads parquet files file-by-file (memory-efficient),
aggregates 30-min binned features across App-usage, Device-usage, and Motion sensors,
and writes a merged CSV: processed_data/sensor_features_halfhour_v2.csv

Expected config.py exports:
 - APP_FOLDER, MUSIC_FOLDER, HEADSET_FOLDER, NOTIF_FOLDER, OUTPUT_DIR

Optional (if present in config):
 - DEVICE_FOLDER (base folder for device usage parquet files)
 - AIRPLANE_FOLDER, BATTERYCHARGE_FOLDER, BATTERYMONITOR_FOLDER, DOZE_FOLDER,
   RINGMODE_FOLDER, SCREEN_FOLDER, TOUCH_FOLDER, USERPRESENCE_FOLDER
 - MOTION_FOLDER, ACTIVITIES_FOLDER, STEPDETECTOR_FOLDER

This final version:
 - DOES NOT produce any rolling or time-of-day derived features
 - DOES NOT include placeholder variance/mean columns (app_event_var_2h, music_play_mean_2h, headset_plug_var_2h)
 - DOES NOT extract motion_running_count or motion_walking_count (both removed)
"""
import os
import glob
import pandas as pd
import numpy as np
from typing import Optional, Iterable

# -------------------------
# Config import (user config)
# -------------------------
from config import APP_FOLDER, HEADSET_FOLDER, MUSIC_FOLDER, NOTIF_FOLDER, OUTPUT_DIR

# optional imports (DEVICE/MOTION)
try:
    from config import DEVICE_FOLDER
except Exception:
    DEVICE_FOLDER = None

# optional explicit device/motion paths (if provided in config)
for name in [
    "AIRPLANE_FOLDER", "BATTERYCHARGE_FOLDER", "BATTERYMONITOR_FOLDER",
    "DOZE_FOLDER", "RINGMODE_FOLDER", "SCREEN_FOLDER", "TOUCH_FOLDER", "USERPRESENCE_FOLDER",
    "MOTION_FOLDER", "ACTIVITIES_FOLDER", "STEPDETECTOR_FOLDER"
]:
    globals().setdefault(name, None)
    try:
        val = __import__("config").__dict__.get(name, None)
        if val is not None:
            globals()[name] = val
    except Exception:
        pass

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# Utilities
# -------------------------
def iter_parquet_files(path_or_file: Optional[str]) -> Iterable[str]:
    """
    Yield parquet file paths. Accept:
      - explicit file path to a single .parquet
      - directory containing many parquet files (recursive search)
      - None -> yields nothing
    """
    if not path_or_file:
        return
    # if explicit file provided
    if os.path.isfile(path_or_file) and path_or_file.lower().endswith(".parquet"):
        yield path_or_file
        return
    # directory provided
    if os.path.isdir(path_or_file):
        patterns = ["**/*.parquet", "*.parquet"]
        for pat in patterns:
            for p in glob.glob(os.path.join(path_or_file, pat), recursive=True):
                yield p
        return
    # maybe path ends with .parquet but is actually a directory
    maybe_dir = os.path.splitext(path_or_file)[0]
    if os.path.isdir(maybe_dir):
        for p in glob.glob(os.path.join(maybe_dir, "**/*.parquet"), recursive=True):
            yield p

def read_parquet_file_as_df(path: str) -> pd.DataFrame:
    """Read a single parquet file into a DataFrame (pandas)."""
    return pd.read_parquet(path)

def make_bin(df: pd.DataFrame, time_col: str = "timestamp") -> pd.DataFrame:
    """
    Convert `time_col` to datetime, drop NaT, add `bin` column floored to 30 minutes.
    This function modifies and returns a small per-file dataframe (safe).
    """
    if time_col not in df.columns:
        raise ValueError(f"make_bin: {time_col} not in dataframe columns")
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col])
    if df.empty:
        return df
    df["bin"] = df[time_col].dt.floor("30min")
    return df

def safe_fillna_zero(df: pd.DataFrame) -> pd.DataFrame:
    """Fill numeric NaNs with 0. Non-numeric kept unchanged."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    return df

def entropy_from_counts(counts: Iterable[float], base: Optional[float] = None) -> float:
    arr = np.asarray(list(counts), dtype=float)
    s = arr.sum()
    if s <= 0:
        return 0.0
    p = arr / s
    res = -np.sum(p * np.log(p + 1e-12))
    if base is not None and base != np.e:
        res = res / np.log(base)
    return float(res)

# -------------------------
# Path resolution helpers
# -------------------------
def infer_device_folder_from_app(app_path: str) -> Optional[str]:
    if not app_path:
        return None
    dirname = os.path.dirname(app_path)
    if os.path.basename(dirname).lower().startswith("app-usage"):
        parent = os.path.dirname(dirname)
        candidate = os.path.join(parent, "Device-usage")
        if os.path.isdir(candidate):
            return candidate
    maybe = os.path.join(os.path.dirname(dirname), "Device-usage")
    if os.path.isdir(maybe):
        return maybe
    return None

def infer_motion_folder_from_app(app_path: str) -> Optional[str]:
    if not app_path:
        return None
    dirname = os.path.dirname(app_path)
    parent = os.path.dirname(dirname)
    candidate = os.path.join(parent, "Motion")
    if os.path.isdir(candidate):
        return candidate
    return None

_effective_device_folder = DEVICE_FOLDER or infer_device_folder_from_app(APP_FOLDER)
_effective_motion_folder = globals().get("MOTION_FOLDER") or infer_motion_folder_from_app(APP_FOLDER)

# -------------------------
# App-usage extractors (streaming)
# -------------------------
def extract_application_features_stream() -> pd.DataFrame:
    """
    Returns columns:
    userid, bin, app_event_count, unique_apps_count, app_usage_entropy
    (No app_event_var_2h)
    """
    print("  - streaming application files...")
    per_app_chunks = []
    for p in iter_parquet_files(APP_FOLDER):
        try:
            df = read_parquet_file_as_df(p)
        except Exception as e:
            print(f"    [app] skipping {p}: {e}")
            continue
        if df.empty:
            continue
        try:
            df = make_bin(df, time_col="timestamp")
        except Exception:
            continue
        if df.empty or "applicationname" not in df.columns:
            continue
        g = df.groupby(["userid", "bin", "applicationname"]).size().reset_index(name="cnt")
        per_app_chunks.append(g)

    if not per_app_chunks:
        return pd.DataFrame(columns=["userid", "bin", "app_event_count", "unique_apps_count", "app_usage_entropy"])

    app_counts = pd.concat(per_app_chunks, ignore_index=True)
    app_counts = app_counts.groupby(["userid", "bin", "applicationname"], as_index=False)["cnt"].sum()

    total = app_counts.groupby(["userid", "bin"], as_index=False)["cnt"].sum().rename(columns={"cnt": "app_event_count"})
    unique_apps = app_counts.groupby(["userid", "bin"], as_index=False)["applicationname"].nunique().rename(columns={"applicationname": "unique_apps_count"})
    entropy_df = app_counts.groupby(["userid", "bin"])["cnt"].apply(lambda s: entropy_from_counts(s.values)).reset_index(name="app_usage_entropy")

    feat = total.merge(unique_apps, on=["userid", "bin"], how="left").merge(entropy_df, on=["userid", "bin"], how="left")
    feat = safe_fillna_zero(feat)
    return feat

def extract_music_features_stream() -> pd.DataFrame:
    """Returns userid, bin, music_play_count """
    print("  - streaming music files...")
    chunks = []
    for p in iter_parquet_files(MUSIC_FOLDER):
        try:
            df = read_parquet_file_as_df(p)
        except Exception as e:
            print(f"    [music] skipping {p}: {e}")
            continue
        if df.empty:
            continue
        try:
            df = make_bin(df, time_col="timestamp")
        except Exception:
            continue
        g = df.groupby(["userid", "bin"]).size().reset_index(name="music_play_count")
        chunks.append(g)
    if not chunks:
        return pd.DataFrame(columns=["userid", "bin", "music_play_count"])
    df = pd.concat(chunks, ignore_index=True).groupby(["userid", "bin"], as_index=False)["music_play_count"].sum()
    return df

def extract_headset_features_stream() -> pd.DataFrame:
    """Returns userid, bin, headset_plug_count """
    print("  - streaming headset files...")
    chunks = []
    for p in iter_parquet_files(HEADSET_FOLDER):
        try:
            df = read_parquet_file_as_df(p)
        except Exception as e:
            print(f"    [headset] skipping {p}: {e}")
            continue
        if df.empty:
            continue
        try:
            df = make_bin(df, time_col="timestamp")
        except Exception:
            continue
        g = df.groupby(["userid", "bin"]).size().reset_index(name="headset_plug_count")
        chunks.append(g)
    if not chunks:
        return pd.DataFrame(columns=["userid", "bin", "headset_plug_count"])
    df = pd.concat(chunks, ignore_index=True).groupby(["userid", "bin"], as_index=False)["headset_plug_count"].sum()
    return df

def extract_notification_features_stream() -> pd.DataFrame:
    """Returns userid, bin, notif_posted_count, notif_unique_packages, notif_entropy"""
    print("  - streaming notification files...")
    per_pkg = []
    per_bin = []
    for p in iter_parquet_files(NOTIF_FOLDER):
        try:
            df = read_parquet_file_as_df(p)
        except Exception as e:
            print(f"    [notif] skipping {p}: {e}")
            continue
        if df.empty:
            continue
        try:
            df = make_bin(df, time_col="timestamp")
        except Exception:
            continue
        per_bin.append(df.groupby(["userid", "bin"]).size().reset_index(name="notif_posted_count"))
        if "package" in df.columns:
            per_pkg.append(df.groupby(["userid", "bin", "package"]).size().reset_index(name="cnt_pkg"))
    if not per_bin:
        return pd.DataFrame(columns=["userid", "bin", "notif_posted_count", "notif_unique_packages", "notif_entropy"])
    notif_bin = pd.concat(per_bin, ignore_index=True).groupby(["userid", "bin"], as_index=False)["notif_posted_count"].sum()
    if per_pkg:
        pkg = pd.concat(per_pkg, ignore_index=True).groupby(["userid", "bin", "package"], as_index=False)["cnt_pkg"].sum()
        unique_pkgs = pkg.groupby(["userid", "bin"], as_index=False)["package"].nunique().rename(columns={"package": "notif_unique_packages"})
        notif_entropy = pkg.groupby(["userid", "bin"])["cnt_pkg"].apply(lambda s: entropy_from_counts(s.values)).reset_index(name="notif_entropy")
        feat = notif_bin.merge(unique_pkgs, on=["userid", "bin"], how="left").merge(notif_entropy, on=["userid", "bin"], how="left")
    else:
        feat = notif_bin.copy()
        feat["notif_unique_packages"] = 0
        feat["notif_entropy"] = 0.0
    feat = safe_fillna_zero(feat)
    return feat

# -------------------------
# Generic device helpers
# -------------------------
def _get_device_path(name: str) -> Optional[str]:
    candidate_names = {
        "airplanemode": "AIRPLANE_FOLDER",
        "batterycharge": "BATTERYCHARGE_FOLDER",
        "batterymonitoringlog": "BATTERYMONITOR_FOLDER",
        "doze": "DOZE_FOLDER",
        "ringmode": "RINGMODE_FOLDER",
        "screen": "SCREEN_FOLDER",
        "touch": "TOUCH_FOLDER",
        "userpresence": "USERPRESENCE_FOLDER",
    }
    cfg_var = candidate_names.get(name)
    if cfg_var and globals().get(cfg_var):
        return globals().get(cfg_var)
    if _effective_device_folder:
        p = os.path.join(_effective_device_folder, f"{name}.parquet")
        if os.path.exists(p):
            return p
        if os.path.isdir(os.path.join(_effective_device_folder, name)):
            return os.path.join(_effective_device_folder, name)
    return None

def _count_true_status_stream(path_or_file: Optional[str], status_col: str = "status", out_col: str = "count_true") -> pd.DataFrame:
    chunks = []
    if not path_or_file:
        return pd.DataFrame(columns=["userid", "bin", out_col])
    for p in iter_parquet_files(path_or_file):
        try:
            df = read_parquet_file_as_df(p)
        except Exception as e:
            print(f"    [device:{out_col}] skipping {p}: {e}")
            continue
        if df.empty or status_col not in df.columns:
            continue
        try:
            df = make_bin(df, time_col="timestamp")
        except Exception:
            continue
        ser = df[status_col].astype(str).str.lower()
        df["_status_bool"] = ser.isin(["true", "1", "t", "yes", "on"])
        g = df.groupby(["userid", "bin"])["_status_bool"].sum().reset_index(name=out_col)
        chunks.append(g)
    if not chunks:
        return pd.DataFrame(columns=["userid", "bin", out_col])
    df = pd.concat(chunks, ignore_index=True).groupby(["userid", "bin"], as_index=False)[out_col].sum()
    return df

def _count_events_stream(path_or_file: Optional[str], out_col: str = "count") -> pd.DataFrame:
    chunks = []
    if not path_or_file:
        return pd.DataFrame(columns=["userid", "bin", out_col])
    for p in iter_parquet_files(path_or_file):
        try:
            df = read_parquet_file_as_df(p)
        except Exception as e:
            print(f"    [device:{out_col}] skipping {p}: {e}")
            continue
        if df.empty:
            continue
        try:
            df = make_bin(df, time_col="timestamp")
        except Exception:
            continue
        g = df.groupby(["userid", "bin"]).size().reset_index(name=out_col)
        chunks.append(g)
    if not chunks:
        return pd.DataFrame(columns=["userid", "bin", out_col])
    df = pd.concat(chunks, ignore_index=True).groupby(["userid", "bin"], as_index=False)[out_col].sum()
    return df

# -------------------------
# Device extractors
# -------------------------
def extract_airplanemode_features_stream() -> pd.DataFrame:
    print("  - streaming airplanemode...")
    path = globals().get("AIRPLANE_FOLDER") or _get_device_path("airplanemode")
    return _count_true_status_stream(path, status_col="status", out_col="airplane_on_count")

def extract_batterycharge_features_stream() -> pd.DataFrame:
    print("  - streaming batterycharge...")
    path = globals().get("BATTERYCHARGE_FOLDER") or _get_device_path("batterycharge")
    return _count_true_status_stream(path, status_col="status", out_col="battery_charge_on_count")

def extract_batterymonitoring_features_stream() -> pd.DataFrame:
    print("  - streaming batterymonitoringlog...")
    path = globals().get("BATTERYMONITOR_FOLDER") or _get_device_path("batterymonitoringlog")
    chunks = []
    if not path:
        return pd.DataFrame(columns=["userid", "bin", "battery_level_mean"])
    for p in iter_parquet_files(path):
        try:
            df = read_parquet_file_as_df(p)
        except Exception as e:
            print(f"    [battery_mon] skipping {p}: {e}")
            continue
        if df.empty or "level" not in df.columns:
            continue
        try:
            df = make_bin(df, time_col="timestamp")
        except Exception:
            continue
        g = df.groupby(["userid", "bin"])["level"].mean().reset_index(name="battery_level_mean")
        chunks.append(g)
    if not chunks:
        return pd.DataFrame(columns=["userid", "bin", "battery_level_mean"])
    df = pd.concat(chunks, ignore_index=True).groupby(["userid", "bin"], as_index=False)["battery_level_mean"].mean()
    return df

def extract_doze_features_stream() -> pd.DataFrame:
    print("  - streaming doze...")
    path = globals().get("DOZE_FOLDER") or _get_device_path("doze")
    return _count_true_status_stream(path, status_col="status", out_col="doze_on_count")

def extract_ringmode_features_stream() -> pd.DataFrame:
    print("  - streaming ringmode...")
    path = globals().get("RINGMODE_FOLDER") or _get_device_path("ringmode")
    chunks = []
    if not path:
        cols = ["userid", "bin", "ring_mode_normal_count", "ring_mode_silent_count", "ring_mode_vibrate_count"]
        return pd.DataFrame(columns=cols)
    for p in iter_parquet_files(path):
        try:
            df = read_parquet_file_as_df(p)
        except Exception as e:
            print(f"    [ringmode] skipping {p}: {e}")
            continue
        if df.empty or "status" not in df.columns:
            continue
        try:
            df = make_bin(df, time_col="timestamp")
        except Exception:
            continue
        df["status_str"] = df["status"].astype(str).str.lower()
        def mode_label(s: str) -> str:
            if "silent" in s:
                return "mode_silent"
            if "vibrate" in s:
                return "mode_vibrate"
            if "normal" in s:
                return "mode_normal"
            return s
        df["mode"] = df["status_str"].apply(mode_label)
        g = df.groupby(["userid", "bin", "mode"]).size().reset_index(name="cnt")
        chunks.append(g)
    if not chunks:
        cols = ["userid", "bin", "ring_mode_normal_count", "ring_mode_silent_count", "ring_mode_vibrate_count"]
        return pd.DataFrame(columns=cols)
    df = pd.concat(chunks, ignore_index=True).groupby(["userid", "bin", "mode"], as_index=False)["cnt"].sum()
    pivot = df.pivot_table(index=["userid", "bin"], columns="mode", values="cnt", fill_value=0).reset_index()
    mapping = {"mode_normal": "ring_mode_normal_count", "mode_silent": "ring_mode_silent_count", "mode_vibrate": "ring_mode_vibrate_count"}
    pivot = pivot.rename(columns=mapping)
    for required in mapping.values():
        if required not in pivot.columns:
            pivot[required] = 0
    return pivot[["userid", "bin", "ring_mode_normal_count", "ring_mode_silent_count", "ring_mode_vibrate_count"]].fillna(0)

def extract_screen_features_stream() -> pd.DataFrame:
    print("  - streaming screen...")
    path = globals().get("SCREEN_FOLDER") or _get_device_path("screen")
    chunks = []
    if not path:
        return pd.DataFrame(columns=["userid", "bin", "screen_on_count"])
    for p in iter_parquet_files(path):
        try:
            df = read_parquet_file_as_df(p)
        except Exception as e:
            print(f"    [screen] skipping {p}: {e}")
            continue
        if df.empty or "status" not in df.columns:
            continue
        try:
            df = make_bin(df, time_col="timestamp")
        except Exception:
            continue
        df["_is_on"] = df["status"].astype(str).str.upper().eq("SCREEN_ON")
        g = df.groupby(["userid", "bin"])["_is_on"].sum().reset_index(name="screen_on_count")
        chunks.append(g)
    if not chunks:
        return pd.DataFrame(columns=["userid", "bin", "screen_on_count"])
    df = pd.concat(chunks, ignore_index=True).groupby(["userid", "bin"], as_index=False)["screen_on_count"].sum()
    return df

def extract_touch_features_stream() -> pd.DataFrame:
    print("  - streaming touch...")
    path = globals().get("TOUCH_FOLDER") or _get_device_path("touch")
    return _count_events_stream(path, out_col="touch_count")

def extract_userpresence_features_stream() -> pd.DataFrame:
    print("  - streaming userpresence...")
    path = globals().get("USERPRESENCE_FOLDER") or _get_device_path("userpresence")
    return _count_true_status_stream(path, status_col="status", out_col="user_present_count")

# -------------------------
# Motion extractors (running & walking removed)
# -------------------------
def extract_activity_features_stream() -> pd.DataFrame:
    print("  - streaming activities...")
    path = globals().get("ACTIVITIES_FOLDER") or (os.path.join(_effective_motion_folder, "activities.parquet") if _effective_motion_folder else None)
    chunks = []
    # expected output columns (running & walking removed)
    expected_cols = [
        "userid", "bin",
        "motion_invehicle_count", "motion_onbicycle_count", "motion_onfoot_count",
        "motion_still_count", "motion_unknown_count", "motion_tilting_count"
    ]
    if not path:
        return pd.DataFrame(columns=expected_cols)
    for p in iter_parquet_files(path):
        try:
            df = read_parquet_file_as_df(p)
        except Exception as e:
            print(f"    [activity] skipping {p}: {e}")
            continue
        if df.empty:
            continue
        if "accuracy" not in df.columns or "label" not in df.columns:
            continue
        # keep high-confidence only
        df = df[df["accuracy"].astype(float) > 90]
        if df.empty:
            continue
        try:
            df = make_bin(df, time_col="timestamp")
        except Exception:
            continue
        df["label"] = df["label"].astype(str).str.lower().str.strip()
        # valid mapping excludes 'running' and 'walking'
        valid = {
            "invehicle": "motion_invehicle_count",
            "onbicycle": "motion_onbicycle_count",
            "onfoot": "motion_onfoot_count",
            "still": "motion_still_count",
            "unknown": "motion_unknown_count",
            "tilting": "motion_tilting_count"
        }
        df = df[df["label"].isin(valid.keys())]
        if df.empty:
            continue
        g = df.groupby(["userid", "bin", "label"]).size().reset_index(name="cnt")
        chunks.append(g)
    if not chunks:
        return pd.DataFrame(columns=expected_cols)
    df = pd.concat(chunks, ignore_index=True)
    df = df.groupby(["userid", "bin", "label"], as_index=False)["cnt"].sum()
    pivot = df.pivot_table(index=["userid", "bin"], columns="label", values="cnt", fill_value=0).reset_index()
    pivot = pivot.rename(columns={
        "invehicle": "motion_invehicle_count",
        "onbicycle": "motion_onbicycle_count",
        "onfoot": "motion_onfoot_count",
        "still": "motion_still_count",
        "unknown": "motion_unknown_count",
        "tilting": "motion_tilting_count"
    })
    # ensure all expected motion columns exist (those left after removal)
    for col in expected_cols[2:]:
        if col not in pivot.columns:
            pivot[col] = 0
    return pivot[["userid", "bin"] + expected_cols[2:]].fillna(0)

def extract_stepdetector_features_stream() -> pd.DataFrame:
    print("  - streaming stepdetector...")
    path = globals().get("STEPDETECTOR_FOLDER") or (os.path.join(_effective_motion_folder, "stepdetector.parquet") if _effective_motion_folder else None)
    if not path:
        return pd.DataFrame(columns=["userid", "bin", "step_count"])
    chunks = []
    for p in iter_parquet_files(path):
        try:
            df = read_parquet_file_as_df(p)
        except Exception as e:
            print(f"    [steps] skipping {p}: {e}")
            continue
        if df.empty:
            continue
        try:
            df = make_bin(df, time_col="timestamp")
        except Exception:
            continue
        g = df.groupby(["userid", "bin"]).size().reset_index(name="step_count")
        chunks.append(g)
    if not chunks:
        return pd.DataFrame(columns=["userid", "bin", "step_count"])
    df = pd.concat(chunks, ignore_index=True).groupby(["userid", "bin"], as_index=False)["step_count"].sum()
    return df

# -------------------------
# Combine and write CSV (no rolling, no time-derived features)
# -------------------------
def combine_features_stream():
    print("🔹 Extracting advanced behavioral features (streaming, low-memory)...")

    # app/music/headset/notif
    app = extract_application_features_stream()
    music = extract_music_features_stream()
    headset = extract_headset_features_stream()
    notif = extract_notification_features_stream()

    # device
    airplane = extract_airplanemode_features_stream()
    battery_charge = extract_batterycharge_features_stream()
    battery_mon = extract_batterymonitoring_features_stream()
    doze = extract_doze_features_stream()
    ring = extract_ringmode_features_stream()
    screen = extract_screen_features_stream()
    touch = extract_touch_features_stream()
    userpresence = extract_userpresence_features_stream()

    # motion
    activity = extract_activity_features_stream()
    steps = extract_stepdetector_features_stream()

    dfs = [
        app, music, headset, notif,
        airplane, battery_charge, battery_mon, doze,
        ring, screen, touch, userpresence,
        activity, steps
    ]

    # Merge all outer on (userid, bin)
    merged = None
    for d in dfs:
        if d is None or d.empty:
            continue
        if merged is None:
            merged = d.copy()
        else:
            merged = merged.merge(d, on=["userid", "bin"], how="outer")

    if merged is None:
        print("No features extracted. Exiting.")
        return

    # Fill numeric NaNs with 0; final fallback fill 0 for anything left
    merged = safe_fillna_zero(merged)
    merged = merged.fillna(0)

    save_path = os.path.join(OUTPUT_DIR, "sensor_features_halfhour_v2.csv")
    merged.to_csv(save_path, index=False)
    print(f" Final feature table saved to: {save_path}")
    print(f" Final feature table shape: {merged.shape}")

if __name__ == "__main__":
    combine_features_stream()
