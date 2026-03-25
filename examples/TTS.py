import wave
import os
import struct
from piper.voice import PiperVoice
import simpleaudio as sa


model_path = os.path.expanduser("en_US-amy-medium.onnx")
voice = PiperVoice.load(model_path)

text = """This full-shot image captures a group of five young individuals—three women and two men—posing for a photo in front of a bright lime-green wall adorned with strings of twinkling fairy lights. The date "09.18.2025 22:01" is visible in the bottom right corner, suggesting it was taken on September 18, 2025, at 10:01 PM.

From left to right:

The first woman on the left has long, straight, light brown hair and is wearing a dark green, sequined mini-dress with thin straps and a short slit on the left side. She has a subtle smile and a light-colored necklace. On her left wrist, she wears a light blue bracelet. She is wearing white athletic socks and black and white athletic shoes.

Next to her is a woman with dark hair, styled in soft waves, and a radiant smile. She’s wearing a strapless, form-fitting orange mini-dress. She has a thin silver necklace and is wearing light-colored open-toed sandals with a strap across the toes and around the ankle.

In the center is another woman with long, wavy blonde hair. She is smiling broadly with her eyes almost closed. She's dressed in a light lavender or periwinkle dress with intricate lace detailing on the bodice and a sweetheart neckline. She also has a delicate gold necklace around her neck.

Behind these three women, slightly to the right, stands a man with dark skin and short, dark hair. He is wearing a long-sleeved, collared black shirt with buttons down the front. He has a neutral expression on his face.

The man on the far right has spiky, highlighted dark hair and is wearing a white long-sleeved button-up shirt with dark pants or jeans. He is smiling and gesturing a peace sign with his right hand. He also appears to be wearing a dark-colored watch or bracelet on his left wrist.

In the foreground, there's a round table draped with a black tablecloth. On the table are several clear plastic water bottles with white and blue labels, a few clear plastic cups, and some black decorative items which resemble small buildings or cityscapes. An LG brand flat-screen TV is mounted on the wall above the group, slightly to the left from the center.

The overall lighting is somewhat dim, which enhances the glow of the fairy lights, creating a festive or celebratory atmosphere. The green wall and casual yet dressed-up attire of the individuals suggest a social gathering or a party."""

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