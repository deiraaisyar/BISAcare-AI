from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import soundfile as sf
import librosa  

MODEL_ID = "ayaayaa/whisper-finetuned-id"
processor = WhisperProcessor.from_pretrained(MODEL_ID)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def transcribe_audio(file_path: str) -> str:
    # load audio (apapun sample rate awalnya)
    audio, sr = sf.read(file_path)

    # resample ke 16000
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000

    # preprocessing
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt").to(device)

    # inference
    with torch.no_grad():
        predicted_ids = model.generate(inputs["input_features"])

    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    return transcription
