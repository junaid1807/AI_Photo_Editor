"""
Phase 1 - Step 1: Voice -> Text
Records audio from mic (or loads a .wav file) and transcribes it with Whisper.
"""

import whisper
import sounddevice as sd
from scipy.io.wavfile import write

MODEL_SIZE = "base"  # tiny | base | small | medium | large  (bigger = slower, more accurate)
SAMPLE_RATE = 16000


def record_audio(filename="audio/command.wav", duration=4):
    """Record `duration` seconds of audio from the default microphone."""
    print(f"🎙️  Recording for {duration}s... speak now.")
    recording = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
    sd.wait()
    write(filename, SAMPLE_RATE, recording)
    print(f"✅ Saved recording to {filename}")
    return filename


def transcribe(filepath: str, model_size: str = MODEL_SIZE) -> str:
    """Transcribe an audio file to text using Whisper."""
    model = whisper.load_model(model_size)
    result = model.transcribe(filepath)
    text = result["text"].strip()
    print(f"📝 Transcribed: {text}")
    return text


if __name__ == "__main__":
    # Quick manual test: record 4s, then transcribe
    path = record_audio()
    command_text = transcribe(path)
    print("COMMAND:", command_text)