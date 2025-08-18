import os
from features.data_asuransi_ai.scan_data import extract_text
import whisper
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_diagnosis(image_bytes=None, text=None, audio_path=None):
    """
    Proses diagnosis dokter dari foto (OCR), text manual, atau audio (speech-to-text).
    """
    if image_bytes is not None:
        # Proses OCR dari gambar
        ocr_result = extract_text(image_bytes)
        return {"jenis": "foto", "hasil": ocr_result}

    if text is not None:
        # Proses text manual
        return {"jenis": "text", "hasil": text}

    if audio_path is not None:
        # Proses audio dengan OpenAI Whisper (local)
        try:
            logger.info(f"Audio file path received: {audio_path}")
            if not audio_path or not isinstance(audio_path, str) or not os.path.exists(audio_path):
                raise Exception(f"Audio file not found or path invalid: {audio_path}")
            file_size = os.path.getsize(audio_path)
            logger.info(f"File size: {file_size} bytes")
            if file_size == 0:
                raise Exception("Audio file is empty")
            model = whisper.load_model("base")
            result = model.transcribe(audio_path, language="indonesian")
            transcription = result["text"]
            logger.info(f"OpenAI Whisper transcription result: '{transcription}'")
            return {"jenis": "audio", "hasil": transcription.strip()}
        except Exception as e:
            logger.error(f"Error in OpenAI Whisper transcribe_audio: {str(e)}")
            return {"jenis": "audio", "hasil": f"Error transcribing audio: {str(e)}"}

    # Jika tidak ada input
    return {"jenis": "none", "hasil": "Tidak ada data diagnosis dokter yang diberikan."}