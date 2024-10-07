import subprocess
import os

# List of scripts to run
scripts = [
    "station4_swc_5.py",
    "station4_swc_10.py",
    "station4_swc_20.py",
    "station4_swc_50.py",
    "station5_swc_5.py",
    "station5_swc_10.py",
    "station5_swc_20.py",
    "station5_swc_50.py"
]

# Directory containing the scripts
script_dir = "Analysis/station-data-analysis"

# Run each script
for script in scripts:
    print(f"Running {script}...")
    script_path = os.path.join(script_dir, script)
    try:
        subprocess.run(["python", script_path], check=True)
        print(f"Finished running {script}")
    except subprocess.CalledProcessError as e:
        print(f"Error running {script}: {e}")
    print("\n" + "="*50 + "\n")

print("All analyses completed.")
