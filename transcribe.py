import whisper
import os
from datetime import datetime
import spacy

os.environ["FFMPEG_BINARY"] = r"C:\Users\NITHISHVARAN T P\Downloads\ffmpeg-7.1.1-essentials_build\ffmpeg-7.1.1-essentials_build\bin\ffmpeg.exe"

def transcribe_audio(audio_file):
    model = whisper.load_model("base")
    result = model.transcribe(audio_file, fp16=False)
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
