import os
import argparse
from datetime import datetime

from personality.personality_ai import generate_personality_reply
from voice_cloning.voice_clone import clone_voice
from lipsync.lipsync import generate_lipsync_video


def run_pipeline(user_text, avatar_image_path, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)

    print("=== Emotional AI Avatar Pipeline ===")

    # 1) Personality + Emotion (Text -> Text)
    print("[1/3] Generating personality-based response...")
    response_text = generate_personality_reply(user_text)
    print("Deceased Response:\n", response_text)

    # 2) Voice (Text -> Audio)
    print("[2/3] Generating voice audio...")
    audio_out = os.path.join(output_dir, "generated_audio.wav")
    clone_voice(response_text, audio_out)
    print("Audio saved at:", audio_out)

    # 3) Lip Sync (Audio + Image -> Video)
    print("[3/3] Generating lip-synced video...")
    lipsync_out_dir = os.path.join(output_dir, "lipsync_results")
    result_dir = generate_lipsync_video(
        image_path=avatar_image_path,
        audio_path=audio_out,
        output_dir=lipsync_out_dir
    )

    print("\n✅ Pipeline completed successfully.")
    print("Lip-sync results directory:", result_dir)
    return result_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Emotional AI Avatar Pipeline")
    parser.add_argument("--text", type=str, required=True, help="User input text")
    parser.add_argument("--image", type=str, required=True, help="Path to avatar image")
    parser.add_argument("--out", type=str, default="outputs", help="Output directory")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Avatar image not found: {args.image}")

    run_pipeline(
        user_text=args.text,
        avatar_image_path=args.image,
        output_dir=args.out
    )
