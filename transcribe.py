import whisper
import os
import numpy as np
import noisereduce as nr
import scipy.io.wavfile as wav
import soundfile as sf
from datetime import datetime
import spacy

os.environ["FFMPEG_BINARY"] = r"C:\Users\NITHISHVARAN T P\Downloads\ffmpeg-7.1.1-essentials_build\ffmpeg-7.1.1-essentials_build\bin\ffmpeg.exe"

def reduce_noise(input_wav, output_wav="denoised_output.wav", noise_duration_sec=1):
    rate, data = wav.read(input_wav)
    if len(data.shape) == 2:
        data = data.mean(axis=1)
    data = data.astype(np.float32)
    noise_sample = data[:int(rate * noise_duration_sec)]
    reduced_noise = nr.reduce_noise(y=data, y_noise=noise_sample, sr=rate)
    sf.write(output_wav, reduced_noise, rate)
    return output_wav

def transcribe_audio(audio_file):
    print("🔉 Reducing noise...")
    clean_audio = reduce_noise(audio_file)

    print("🤖 Transcribing with Whisper...")
    model = whisper.load_model("base")
    result = model.transcribe(clean_audio, fp16=False)
    return result["text"]

def save_transcript(text):
    os.makedirs("transcripts", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_path = f"transcripts/{timestamp}.txt"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\n✅ Transcript saved to: {file_path}")

def extract_keywords(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    keywords = {chunk.text.strip() for chunk in doc.noun_chunks if len(chunk.text.strip()) > 2}
    print("\n🧠 Extracted Keywords:")
    for word in keywords:
        print(f"- {word}")

if __name__ == "__main__":
    print("🔊 Loading Whisper model...")
    print("🎙️ Transcribing 'output.wav'...")

    transcript = transcribe_audio("output.wav")

    print("\n📄 Transcription Result:\n", transcript)

    save_transcript(transcript)
    extract_keywords(transcript)
