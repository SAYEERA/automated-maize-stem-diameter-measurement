import os
import subprocess

# ───── CONFIG ─────────────────────────────────────────────
# Initialize hidden root window

base_output_dir = r"test_images/sample_outputs"
yolo_model_path = r"weights/best.pt"

print(f"\nSelected Output Directory: {base_output_dir}")
print(f"\nSelected YOLO Model Path: {yolo_model_path}")

conf_threshold = 0.25

# ───── GET ALL BATCH FOLDERS ─────────────────────────────
all_batches = [
    d for d in os.listdir(base_output_dir)
    if os.path.isdir(os.path.join(base_output_dir, d))
]

# ───── PROCESS EACH BATCH ────────────────────────────────  
    session_path = os.path.join(base_output_dir, batch_folder)
    blackbg_path = os.path.join(session_path, " processed__blackbg", "jpg")
    yolo_output_path = os.path.join(session_path, "yolo_output")

    # ─── Skip if YOLO already run ───
    if os.path.exists(yolo_output_path):
        print(f"YOLO already completed for: {batch_folder}")
        continue

    # ─── Skip if no black background images ───
    image_files = [
        f for f in os.listdir(blackbg_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    if not image_files:
        print(f" No images found in {blackbg_path}. Skipping YOLO for {batch_folder}.")
        continue

    # ─── Build and Run YOLO Command ───
    print(f" Running YOLO on: {batch_folder}")
    yolo_cmd = (
        f"yolo task=detect mode=predict model=\"{yolo_model_path}\" "
        f"source=\"{blackbg_path}\" conf={conf_threshold} save_txt=True "
        f"project=\"{session_path}\" name=\"yolo_output\""
    )

    try:
        subprocess.run(yolo_cmd, check=True, shell=True)
        print(f"YOLO completed for {batch_folder}")
    except subprocess.CalledProcessError as e:
        print(f"YOLO failed for {batch_folder}: {e}")

print("\n All YOLO predictions complete.")

