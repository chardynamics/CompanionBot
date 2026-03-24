import openwakeword
from openwakeword.model import Model

# Instantiate the model(s)
model = Model()

# Get audio data containing 16-bit 16khz PCM audio data from a file, microphone, network stream, etc.
# For the best efficiency and latency, audio frames should be multiples of 80 ms, with longer frames
# increasing overall efficiency at the cost of detection latency
frame = my_function_to_get_audio_frame()

# Get predictions for the frame
prediction = model.predict(frame)

score = list(prediction.values())[0]

    if score > WAKE_THRESHOLD:
        print("Wake word detected!")

        filename = record_until_silence()
        text = transcribe_audio(filename)

        model_return = api_call(text)

        print("Back to listening...\n")