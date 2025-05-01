import sounddevice as sd
from scipy.io.wavfile import write
def record_audio(filename="output.wav", duration=5):
    fs = 44100  
    print("Recording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=2) 
    sd.wait()  
    write(filename, fs, audio)  
    print(f"Recording saved as {filename}")
record_audio("output.wav", 15)
