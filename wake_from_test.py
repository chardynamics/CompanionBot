import pyaudio
import numpy as np
#import openwakeword
import soxr
from openwakeword.model import Model

# Get microphone stream
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 400
audio = pyaudio.PyAudio()
mic_stream = audio.open(format=FORMAT, channels=CHANNELS, rate=44100, input=True, frames_per_buffer=CHUNK)
resampler = soxr.ResampleStream(44100, RATE, CHANNELS, dtype='int16')
buffer = np.zeros(16000, dtype=np.int16)

# Load pre-trained openwakeword models
owwModel = Model(wakeword_models=["./hey_rhasspy_v0.1.onnx"])

# Run capture loop continuosly, checking for wakewords
if __name__ == "__main__":
    # Generate output string header
    print("\n\n")
    print("#"*100)
    print("Listening for wakewords...")
    print("#"*100)

    while True:
        raw_audio = mic_stream.read(CHUNK, exception_on_overflow=False)
        audio_np = np.frombuffer(raw_audio, dtype=np.int16)

        resampled = resampler.resample_chunk(audio_np)
        print("RESAMPLED len:", len(resampled), "max:", np.max(resampled) if len(resampled) else 0)
        if isinstance(resampled, bytes):
            resampled = np.frombuffer(resampled, dtype=np.int16)

        if len(resampled) == 0:
            continue
        
        buffer = np.roll(buffer, -len(resampled))
        buffer[-len(resampled):] = resampled

        prediction = owwModel.predict(buffer[-400:])