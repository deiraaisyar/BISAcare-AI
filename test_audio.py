import soundfile as sf
import librosa
import torch
from transformers import pipeline

# baca audio
audio, sr = sf.read("WhatsApp Audio 2025-08-17 at 23.48.36.mp3")

# resample ke 16k
if sr != 16000:
    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    sr = 16000

pipe = pipeline("automatic-speech-recognition", model="ayaayaa/whisper-finetuned-id")

with torch.no_grad():
    inputs = pipe.feature_extractor(audio, sampling_rate=sr, return_tensors="pt").input_features
    result = pipe.model.generate(inputs)
    text = pipe.tokenizer.batch_decode(result, skip_special_tokens=True)

print(text)
