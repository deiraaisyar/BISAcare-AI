from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
import torch

model_id = "ayaayaa/whisper_finetuning_id"
processor = WhisperProcessor.from_pretrained(model_id)
model = WhisperForConditionalGeneration.from_pretrained(model_id)

# load audio
audio, sr = librosa.load("WhatsApp Audio 2025-08-17 at 23.48.36.mp3", sr=16000)
inputs = processor(audio, sampling_rate=sr, return_tensors="pt")

# generate
with torch.no_grad():
    predicted_ids = model.generate(**inputs)

text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
print("Transkripsi (manual):", text)
