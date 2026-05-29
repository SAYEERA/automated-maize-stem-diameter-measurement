import subprocess
import sys

def run_script(script_name):
    try:
        print(f"\n Running {script_name}...\n")
        result = subprocess.run([sys.executable, script_name], check=True)
        print(f"Finished {script_name}\n")
    except subprocess.CalledProcessError as e:
        print(f"Error while running {script_name}: {e}")
        sys.exit(1)  # Stop further execution if a script fails

if __name__ == "__main__":
    run_script("corn_images_preprocessing")
    run_script("corn_yolo_detection.py")
    run_script("tangent_diameter_measurement.py")
    run_script("horizontal_diameter_measurement")
   
    print("🎉 All Corn scripts completed successfully.")
