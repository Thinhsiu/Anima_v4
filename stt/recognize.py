import sounddevice as sd
import queue
import vosk
import sys
import json

# Path to your Vosk model
MODEL_PATH = "stt/models/vosk-model-small-en-us-0.15"

# Audio settings
samplerate = 16000
device = None  # Use default input device

q = queue.Queue()

def callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))

def recognize():
    model = vosk.Model(MODEL_PATH)
    with sd.RawInputStream(samplerate=samplerate, blocksize=8000, device=device, dtype='int16',
                           channels=1, callback=callback):
        print("🎤 Speak now...")
        rec = vosk.KaldiRecognizer(model, samplerate)
        while True:
            data = q.get()
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                text = result.get("text", "")
                return text
