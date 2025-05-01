import whisper
import os
os.environ["FFMPEG_BINARY"] = r"C:\Users\NITHISHVARAN T P\Downloads\ffmpeg-7.1.1-essentials_build\ffmpeg-7.1.1-essentials_build\bin\ffmpeg.exe"
def transcribe_audio(audio_file):
    model = whisper.load_model("base")
    result = model.transcribe(audio_file, fp16=False)  
    return result["text"]
if __name__ == "__main__":
    print("Loading Whisper model...")
    print("Transcribing 'output.wav'...")
    transcribe = transcribe_audio("output.wav")
    print("Transcription result:", transcribe)
