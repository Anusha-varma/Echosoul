import os
import soundfile as sf
import torch
from transformers import pipeline


class VoiceCloner:
    def __init__(self, device=None):
        if device is None:
            self.device = 0 if torch.cuda.is_available() else -1
        else:
            self.device = device

        self.synthesizer = None
        self._load_model()

    def _load_model(self):
        try:
            print("Loading Bark TTS model...")
            self.synthesizer = pipeline(
                "text-to-speech",
                model="suno/bark-small",
                device=self.device
            )
            print("Loaded suno/bark-small")
        except Exception as e:
            print(f"Failed to load Bark model: {e}")
            print("Falling back to VITS model...")
            self.synthesizer = pipeline(
                "text-to-speech",
                model="espnet/kan-bayashi_ljspeech_vits",
                device=self.device
            )
            print("Loaded espnet VITS model")

    def generate_audio(self, text, output_audio_path):
        print("Generating speech...")
        speech = self.synthesizer(text)

        os.makedirs(os.path.dirname(output_audio_path), exist_ok=True)

        sf.write(
            output_audio_path,
            speech["audio"],
            samplerate=speech["sampling_rate"]
        )

        print(f"Audio saved to: {output_audio_path}")
        return output_audio_path


def clone_voice(text, output_audio_path):
    cloner = VoiceCloner()
    return cloner.generate_audio(text, output_audio_path)


if __name__ == "__main__":
    # Standalone test
    out = clone_voice("I'm always with you", "outputs/generated_audio.wav")
    print("Generated:", out)
