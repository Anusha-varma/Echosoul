import os
import subprocess
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
WAV2LIP_DIR = os.path.join(BASE_DIR, "Easy-Wav2Lip")
CHECKPOINT_DIR = os.path.join(WAV2LIP_DIR, "checkpoints")

def run(cmd):
    print(">>", cmd)
    subprocess.run(cmd, shell=True, check=True)

def setup():
    os.chdir(BASE_DIR)

    if not os.path.exists(WAV2LIP_DIR):
        run("git clone -b v8.3 https://github.com/anothermartz/Easy-Wav2Lip.git")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    ckpt_path = os.path.join(CHECKPOINT_DIR, "wav2lip.pth")
    if not os.path.exists(ckpt_path):
        run(f"curl -L -o {ckpt_path} https://github.com/Rudrabha/Wav2Lip/releases/download/v1.0/wav2lip.pth")

    run("pip install numpy==1.26.4")
    run("pip install torch torchvision torchaudio")
    run("pip install tqdm opencv-python scipy librosa imageio imageio-ffmpeg pyyaml easydict facexlib gfpgan kornia yacs")

    print("✅ LipSync setup completed successfully.")

if __name__ == "__main__":
    setup()
