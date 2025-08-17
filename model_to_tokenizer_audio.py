from transformers import WhisperTokenizer, WhisperFeatureExtractor, WhisperProcessor

base_model = "openai/whisper-small" 

tokenizer = WhisperTokenizer.from_pretrained(base_model, language="indonesian", task="transcribe")
feature_extractor = WhisperFeatureExtractor.from_pretrained(base_model)
processor = WhisperProcessor.from_pretrained(base_model, language="indonesian", task="transcribe")

tokenizer.push_to_hub("ayaayaa/whisper_finetuning_id")
feature_extractor.push_to_hub("ayaayaa/whisper_finetuning_id")
processor.push_to_hub("ayaayaa/whisper_finetuning_id")
