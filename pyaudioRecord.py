import pyaudio
import wave
import soundfile as sf
import soxr

RATE = 44100
CHANNELS = 1
CHUNK = 320

pa = pyaudio.PyAudio()
stream = pa.open(
    rate=RATE,
    channels=CHANNELS,
    format=pyaudio.py,
    input=True,
    frames_per_buffer=CHUNK
)

OUTPUT_FILENAME = "OUTPUT.wav"

for i in range(0, int(RATE / CHUNK * 5)):
    data = stream.read(CHUNK)
    frames.append(data)


# Open input audio file
in_file = sf.SoundFile('OUTPUT.wav', 'r')
source_rate = in_file.samplerate
channels = in_file.channels

# Config ResampleStream
resampler = soxr.ResampleStream(source_rate, RATE, CHANNELS, dtype='paInt16')

# Stop and close the stream
stream.stop_stream()
stream.close()
pa.terminate()  

with wave.open(OUTPUT_FILENAME, 'wb') as wf:
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(pa.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))

print(f"File saved as {OUTPUT_FILENAME}")