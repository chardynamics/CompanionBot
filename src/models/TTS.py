import wave
import os
import struct
from piper.voice import PiperVoice
import simpleaudio as sa


model_path = os.path.expanduser("mv2.onnx")
voice = PiperVoice.load(model_path)

text = """Hi! I'm Alexa, your companion AI robot. Feel free to ask me anything or just chat!"""

output_filename = "output.wav"
sample_rate = 22050

with wave.open(output_filename, 'wb') as wav_file:
    wav_file.setnchannels(1)  # Mono audio
    wav_file.setsampwidth(2)  # 16-bit sample width (S16_LE)
    wav_file.setframerate(sample_rate)

    # Now process your long text
    for audio_chunk in voice.synthesize(text):
        # Use the pre-formatted bytes attribute we found
        wav_file.writeframes(audio_chunk.audio_int16_bytes)

print(f"Audio saved to {output_filename}")

wave_obj = sa.WaveObject.from_wave_file("output.wav")
play_obj = wave_obj.play()
play_obj.wait_done()