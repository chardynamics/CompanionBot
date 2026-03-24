import sounddevice as sd
import numpy as np
import soxr
from openwakeword.model import Model

# Audio settings
INPUT_SR = 44100
TARGET_SR = 16000
CHANNELS = 1
BLOCKSIZE = 1024  # input frames per chunk

# Create streaming resampler
resampler = soxr.ResampleStream(
    INPUT_SR,
    TARGET_SR,
    CHANNELS,
    dtype='float32'
)

# Load wake word model
model = Model()  # or your model

def audio_callback(indata, frames, time, status):
    global buffer

    resampled = resampler.process(audio)
    buffer = np.concatenate([buffer, resampled])

    if status:
        print(status)

    # Convert to mono float32
    audio = indata[:, 0].astype(np.float32)

    # Resample chunk
    resampled = resampler.process(audio)

    # OpenWakeWord expects 16kHz audio
    while len(buffer) >= 1280:
        chunk = buffer[:1280]
        buffer = buffer[1280:]

        prediction = model.predict(chunk)

# Start stream
with sd.InputStream(
    samplerate=INPUT_SR,
    channels=CHANNELS,
    callback=audio_callback,
    blocksize=BLOCKSIZE
):
    print("Listening...")
    while True:
        pass