#!/usr/bin/env python3
"""
Wake word listener for Raspberry Pi 5
- Captures mic audio at 44100 Hz
- Resamples to 16000 Hz using soxr
- Buffers into 400-sample chunks
- Feeds into openWakeWord for detection
 
Dependencies:
    pip install pyaudio soxr openwakeword numpy
    sudo apt-get install portaudio19-dev
"""
 
import pyaudio
import soxr
import numpy as np
from collections import deque
from openwakeword.model import Model
 
# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
MIC_RATE       = 44100      # Hardware capture rate (Hz)
TARGET_RATE    = 16000      # openWakeWord expected rate (Hz)
OWW_CHUNK      = 400        # Samples per chunk at 16kHz (25 ms)
CHANNELS       = 1
FORMAT         = pyaudio.paInt16
 
# How many mic samples we need to produce at least OWW_CHUNK resampled samples.
# Slightly oversized so the buffer always has enough to drain.
# ratio = MIC_RATE / TARGET_RATE  ≈ 2.75625
RESAMPLE_RATIO = MIC_RATE / TARGET_RATE
MIC_CHUNK      = int(np.ceil(OWW_CHUNK * RESAMPLE_RATIO))  # ≈ 1103 samples
 
# openWakeWord detection threshold (0.0 – 1.0)
DETECTION_THRESHOLD = 0.5
 
# ──────────────────────────────────────────────
# Resampler (stateful — preserves continuity across chunks)
# ──────────────────────────────────────────────
resampler = soxr.ResampleStream(
    in_rate=MIC_RATE,
    out_rate=TARGET_RATE,
    num_channels=CHANNELS,
    quality="HQ",           # Options: "QQ", "LQ", "MQ", "HQ", "VHQ"
    dtype="int16",
)
 
# ──────────────────────────────────────────────
# Sample buffer — accumulates resampled audio
# until we have a full OWW_CHUNK worth
# ──────────────────────────────────────────────
sample_buffer: deque[int] = deque()
 
 
def drain_buffer(model: Model) -> None:
    """Pull OWW_CHUNK-sized windows from the buffer and run inference."""
    while len(sample_buffer) >= OWW_CHUNK:
        chunk = np.array(
            [sample_buffer.popleft() for _ in range(OWW_CHUNK)],
            dtype=np.int16,
        )
        predictions = model.predict(chunk)
        for wake_word, score in predictions.items():
            if score >= DETECTION_THRESHOLD:
                print(f"[DETECTED] '{wake_word}'  score={score:.3f}")
 
 
# ──────────────────────────────────────────────
# PyAudio callback (runs on a background thread)
# ──────────────────────────────────────────────
def audio_callback(
    in_data, frame_count, time_info, status_flags, *, oww_model: Model
):
    if status_flags:
        print(f"[WARNING] PyAudio status: {status_flags}")
 
    # 1. Decode raw bytes → int16 numpy array
    mic_audio = np.frombuffer(in_data, dtype=np.int16)
 
    # 2. Resample 44100 → 16000 (stateful — no gaps between callbacks)
    resampled = resampler.resample_chunk(mic_audio, last=False)
 
    # 3. Push resampled samples into the ring buffer
    sample_buffer.extend(resampled.tolist())
 
    # 4. Run inference whenever we have a full chunk
    drain_buffer(oww_model)
 
    return (None, pyaudio.paContinue)
 
 
# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    print("Loading openWakeWord model …")
    # Pass wakeword_models=["path/to/model.tflite"] to load a custom model,
    # or leave empty to use the built-in "hey_jarvis" / "alexa" defaults.
    oww_model = Model(inference_framework="tflite")
    print(f"Loaded models: {list(oww_model.models.keys())}")
 
    pa = pyaudio.PyAudio()
 
    # Show available input devices (handy for debugging on RPi)
    print("\nAvailable input devices:")
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info["maxInputChannels"] > 0:
            print(f"  [{i}] {info['name']}")
 
    stream = pa.open(
        rate=MIC_RATE,
        channels=CHANNELS,
        format=FORMAT,
        input=True,
        frames_per_buffer=MIC_CHUNK,
        stream_callback=lambda in_data, frame_count, time_info, status:
            audio_callback(
                in_data, frame_count, time_info, status, oww_model=oww_model
            ),
    )
 
    print(
        f"\nListening …  "
        f"(mic={MIC_RATE} Hz → target={TARGET_RATE} Hz, "
        f"chunk={OWW_CHUNK} samples = {OWW_CHUNK/TARGET_RATE*1000:.1f} ms)"
    )
    print("Press Ctrl+C to stop.\n")
 
    try:
        stream.start_stream()
        while stream.is_active():
            pass  # Callback runs on PyAudio's internal thread
    except KeyboardInterrupt:
        print("\nStopping …")
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()
        print("Done.")
 
 
if __name__ == "__main__":
    main()