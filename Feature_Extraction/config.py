# =======================================
# config.py
# =======================================
import os

# === Root folders ===
BASE_FOLDER = r"E:\trio_grad\2020-DiversityOne-Trento\Sensors"

APPUSAGE_FOLDER = os.path.join(BASE_FOLDER, "App-usage")
DEVICE_FOLDER   = os.path.join(BASE_FOLDER, "Device-usage")
MOTION_FOLDER   = os.path.join(BASE_FOLDER, "Motion")

DIACHRONIC_FOLDER = r"E:\trio_grad\2020-DiversityOne-Trento\Diachronic-Interactions"

# === Output folder ===
OUTPUT_DIR = "processed_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =======================================
# App-usage sensor subfolders
# =======================================
APP_FOLDER     = os.path.join(APPUSAGE_FOLDER, "application.parquet")
HEADSET_FOLDER = os.path.join(APPUSAGE_FOLDER, "headsetplug.parquet")
MUSIC_FOLDER   = os.path.join(APPUSAGE_FOLDER, "music.parquet")
NOTIF_FOLDER   = os.path.join(APPUSAGE_FOLDER, "notification.parquet")

# =======================================
# Device-usage sensor subfolders
# =======================================
AIRPLANE_FOLDER        = os.path.join(DEVICE_FOLDER, "airplanemode.parquet")
BATTERYCHARGE_FOLDER   = os.path.join(DEVICE_FOLDER, "batterycharge.parquet")
BATTERYMONITOR_FOLDER  = os.path.join(DEVICE_FOLDER, "batterymonitoringlog.parquet")
DOZE_FOLDER            = os.path.join(DEVICE_FOLDER, "doze.parquet")
RINGMODE_FOLDER        = os.path.join(DEVICE_FOLDER, "ringmode.parquet")
SCREEN_FOLDER          = os.path.join(DEVICE_FOLDER, "screen.parquet")
TOUCH_FOLDER           = os.path.join(DEVICE_FOLDER, "touch.parquet")
USERPRESENCE_FOLDER    = os.path.join(DEVICE_FOLDER, "userpresence.parquet")

# =======================================
# Motion sensor subfolders
# =======================================
ACTIVITIES_FOLDER     = os.path.join(MOTION_FOLDER, "activities.parquet")
STEPDETECTOR_FOLDER   = os.path.join(MOTION_FOLDER, "stepdetector.parquet")

# =======================================
# Ground truth (Diachronic)
# =======================================
TIMEDIARIES_PARQUET = os.path.join(DIACHRONIC_FOLDER, "timediaries.parquet")

# === Print paths for confirmation ===
print("✅ Paths configured successfully!")
print("App-usage folder:", APPUSAGE_FOLDER)
print("Device-usage folder:", DEVICE_FOLDER)
print("Motion folder:", MOTION_FOLDER)
print("Diachronic folder:", DIACHRONIC_FOLDER)
print("Output folder:", OUTPUT_DIR)
