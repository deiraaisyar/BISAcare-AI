from transformers import (WhisperTokenizer, WhisperFeatureExtractor,
                          WhisperProcessor, WhisperForConditionalGeneration)

repo_id = "ayaayaa/whisper_finetuning_id"
base_model = "openai/whisper-small"

# 1. Load tokenizer dari pre-trained, tambahkan bahasa & task
tokenizer = WhisperTokenizer.from_pretrained(
    base_model, language="indonesian", task="transcribe"
)

# 2. Load feature extractor dari pre-trained
feature_extractor = WhisperFeatureExtractor.from_pretrained(base_model)

# 3. Gabungkan jadi processor
processor = WhisperProcessor(feature_extractor=feature_extractor,
                             tokenizer=tokenizer)

# 4. Save & push tokenizer + processor ke repo hasil fine-tune
tokenizer.push_to_hub(repo_id)
feature_extractor.push_to_hub(repo_id)
processor.push_to_hub(repo_id)

# 5. Load model fine-tuned kamu
model = WhisperForConditionalGeneration.from_pretrained(repo_id)

# 6. Inference test dengan pipeline
from transformers import pipeline
asr = pipeline("automatic-speech-recognition", model=repo_id, device="cpu")

print(asr("path/to/your/audio.wav", chunk_length_s=30, generate_kwargs={"language":"indonesian","task":"transcribe"}))
