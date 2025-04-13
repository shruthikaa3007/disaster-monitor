import os
import subprocess
import time

# API credentials and model details
API_KEY = "ymIHPOetwVTt9MjhUqsC"
PROJECT_ID = "xview2-xbd"
MODEL_VERSION = "8"
MODEL_ID = f"{PROJECT_ID}/{MODEL_VERSION}"
IMAGE_PATH = "img_1.png"
OUTPUT_DIR = "output"

def start_inference_server():
    print("\nStarting inference server...")
    subprocess.Popen(["inference", "server", "start"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(10)

def run_inference():
    print("\nRunning inference on image:", IMAGE_PATH)

    result = subprocess.run([
        "inference", "infer",
        "--input", IMAGE_PATH,
        "--model_id", MODEL_ID,
        "--api-key", API_KEY,
        "--output_location", OUTPUT_DIR,
        "--visualise"
    ], capture_output=True, text=True, encoding="utf-8")


    print("\nInference Output:\n", result.stdout)
    if result.stderr:
        print("\nError:\n", result.stderr)

# Main function
if __name__ == "__main__":
    start_inference_server()
    run_inference()
