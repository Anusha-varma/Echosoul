import os
import subprocess
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WAV2LIP_DIR = os.path.join(BASE_DIR, "Easy-Wav2Lip")

def generate_lipsync_video(image_path, audio_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(output_dir, timestamp)
    os.makedirs(result_dir, exist_ok=True)

    cmd = f"""
    python "{os.path.join(WAV2LIP_DIR, 'inference.py')}" \
    --driven_audio "{audio_path}" \
    --source_image "{image_path}" \
    --result_dir "{result_dir}" \
    --still \
    --preprocess full
    """

    print("Running lip-sync inference...")
    subprocess.run(cmd, shell=True, check=True)

    return result_dir
