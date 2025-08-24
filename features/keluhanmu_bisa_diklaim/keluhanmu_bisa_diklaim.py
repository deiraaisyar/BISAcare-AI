import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import json
import re
import tempfile
import shutil
import logging
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline
import torch
import librosa

# Setup logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

client = InferenceClient(
    model="meta-llama/Llama-3.2-3B-Instruct",
    token=os.getenv("HF_TOKEN")
)

MEDICAL_ANALYSIS_PROMPT = """Anda adalah asisten AI medis yang membantu menganalisis keluhan kesehatan untuk asuransi.
Berdasarkan keluhan yang diberikan, berikan analisis dalam format JSON:

{
    "persentase_klaim": "angka fix, misal 80, 90, 90.5, 10 (jangan gunakan rentang atau sampai dengan)",
    "kemungkinan_diagnosis": "daftar kemungkinan diagnosis",
    "rekomendasi_tindakan": "saran tindakan medis",
    "tingkat_urgensi": "rendah/sedang/tinggi",
    "dokumen_pendukung": "daftar dokumen yang diperlukan"
}

Berikan jawaban yang akurat dan profesional. Persentase klaim harus berupa satu angka pasti, bukan rentang atau 'sampai dengan'. Contoh: 80, 90, 90.5, 10."""

# Ganti fungsi transcribe_audio agar pakai pipeline Whisper Transformers
pipe = pipeline("automatic-speech-recognition", model="ayaayaa/whisper-finetuned-id")

def transcribe_audio(audio_file_path):
    """
    Convert audio file to text using HuggingFace Whisper pipeline (ayaayaa/whisper-finetuned-id)
    """
    try:
        logger.info(f"Audio file path received: {audio_file_path}")
        if not audio_file_path or not isinstance(audio_file_path, str) or not os.path.exists(audio_file_path):
            raise Exception(f"Audio file not found or path invalid: {audio_file_path}")
        file_size = os.path.getsize(audio_file_path)
        logger.info(f"File size: {file_size} bytes")
        if file_size == 0:
            raise Exception("Audio file is empty")

        import soundfile as sf
        import librosa
        import torch

        # Load audio
        audio, sr = sf.read(audio_file_path)
        # Resample to 16kHz if needed
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000

        # Use Whisper pipeline components directly for robust inference
        with torch.no_grad():
            inputs = pipe.feature_extractor(audio, sampling_rate=sr, return_tensors="pt").input_features
            result = pipe.model.generate(inputs)
            text = pipe.tokenizer.batch_decode(result, skip_special_tokens=True)

        transcription = text[0] if isinstance(text, list) and len(text) > 0 else ""
        logger.info(f"Whisper pipeline transcription result: '{transcription}'")
        return transcription.strip()
    except Exception as e:
        logger.error(f"Error in pipeline transcribe_audio: {str(e)}")
        raise Exception(f"Error transcribing audio: {str(e)}")

def analyze_health_complaint(keluhan_text):
    """
    Menganalisis keluhan kesehatan dan memberikan persentase klaim + diagnosis
    """
    try:
        # Prompt untuk analisis medis
        analysis_prompt = f"""
        Analisis keluhan kesehatan berikut untuk keperluan asuransi:
        
        Keluhan: "{keluhan_text}"
        
        Berikan analisis dalam format JSON dengan:
        1. persentase_klaim (0-100%)
        2. kemungkinan_diagnosis (list)
        3. rekomendasi_tindakan
        4. tingkat_urgensi
        5. dokumen_pendukung_yang_diperlukan
        
        Jawab dalam bahasa Indonesia.
        """
        
        messages = [
            {"role": "system", "content": MEDICAL_ANALYSIS_PROMPT},
            {"role": "user", "content": analysis_prompt}
        ]
        
        response = client.chat_completion(
            messages=messages,
            max_tokens=1024,
            temperature=0.3,
            stream=False
        )
        
        ai_response = response.choices[0].message.content
        
        # Coba parse JSON dari response
        try:
            # Extract JSON dari response jika ada
            json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                # Fallback jika tidak ada JSON
                result = parse_text_response(ai_response, keluhan_text)
        except:
            result = parse_text_response(ai_response, keluhan_text)
            
        return result
        
    except Exception as e:
        # Fallback analysis berdasarkan keyword
        return fallback_analysis(keluhan_text, str(e))

def analyze_health_complaint_from_audio(audio_file_path):
    """
    Menganalisis keluhan kesehatan dari file audio
    """
    try:
        logger.info(f"Analyzing health complaint from audio: {audio_file_path}")
        
        # Transcribe audio to text
        keluhan_text = transcribe_audio(audio_file_path)
        logger.info(f"Successfully transcribed: '{keluhan_text}'")
        
        if not keluhan_text or keluhan_text.strip() == "":
            raise Exception("Transcription is empty")
        
        # Analyze the transcribed text
        result = analyze_health_complaint(keluhan_text)
        
        # Add transcribed text to result
        result["transcribed_text"] = keluhan_text
        
        return result
        
    except Exception as e:
        logger.error(f"Error in analyze_health_complaint_from_audio: {str(e)}")
        return {
            "error": f"Error processing audio: {str(e)}",
            "persentase_klaim": "Tidak dapat ditentukan",
            "kemungkinan_diagnosis": ["Error dalam pemrosesan audio - " + str(e)],
            "rekomendasi_tindakan": [
                "Pastikan file audio tidak rusak",
                "Coba rekam ulang dengan suara yang lebih jelas",
                "Gunakan format MP3 atau WAV",
                "Atau gunakan input text sebagai alternatif"
            ],
            "tingkat_urgensi": "tidak dapat ditentukan",
            "dokumen_pendukung": [],
            "transcribed_text": "",
            "debug_info": str(e)
        }

def parse_text_response(response_text, keluhan):
    """
    Parse response text menjadi format yang diinginkan.
    Jika respons AI mengandung persentase klaim, gunakan itu.
    Jika tidak, fallback ke analisis sederhana.
    """
    # Coba cari angka persentase di response AI
    match = re.search(r'persentase[_\s]?klaim["\':\s]*([0-9]+(?:\.[0-9]+)?)', response_text, re.IGNORECASE)
    if match:
        persentase = float(match.group(1))
        if persentase > 100: persentase = 100
        if persentase < 0: persentase = 0
    else:
        # Fallback sederhana jika AI gagal
        keluhan_lower = keluhan.lower()
        if any(symptom in keluhan_lower for symptom in ['nyeri dada', 'sesak napas', 'demam tinggi', 'muntah darah', 'pingsan', 'stroke']):
            persentase = 90
            urgensi = "tinggi"
        elif any(symptom in keluhan_lower for symptom in ['demam', 'batuk', 'sakit kepala', 'mual', 'diare', 'nyeri perut']):
            persentase = 75
            urgensi = "sedang"
        else:
            persentase = 50
            urgensi = "rendah"

    # Tentukan urgensi dari AI jika ada, jika tidak fallback
    urgensi_match = re.search(r'tingkat[_\s]?urgensi["\':\s]*(rendah|sedang|tinggi)', response_text, re.IGNORECASE)
    urgensi = urgensi_match.group(1).lower() if urgensi_match else "sedang"

    return {
        "persentase_klaim": persentase,
        "kemungkinan_diagnosis": extract_diagnosis_from_text(response_text, keluhan),
        "rekomendasi_tindakan": extract_recommendations_from_text(response_text),
        "tingkat_urgensi": urgensi,
        "dokumen_pendukung": [
            "Surat rujukan dokter",
            "Hasil pemeriksaan laboratorium",
            "Foto rontgen (jika diperlukan)",
            "Resep obat dari dokter"
        ]
    }

def extract_diagnosis_from_text(text, keluhan):
    """Extract kemungkinan diagnosis dari text"""
    keluhan_lower = keluhan.lower()
    
    # Mapping sederhana keluhan ke diagnosis
    diagnosis_mapping = {
        'demam': ['Infeksi virus', 'Infeksi bakteri', 'Flu'],
        'batuk': ['Bronkitis', 'Pneumonia', 'Asma', 'ISPA'],
        'nyeri dada': ['Angina', 'Serangan jantung', 'Refluks asam'],
        'sakit kepala': ['Migrain', 'Tension headache', 'Sinusitis'],
        'mual': ['Gastritis', 'Food poisoning', 'Migrain'],
        'diare': ['Gastroenteritis', 'Food poisoning', 'IBS']
    }
    
    diagnosis_list = []
    for symptom, diagnoses in diagnosis_mapping.items():
        if symptom in keluhan_lower:
            diagnosis_list.extend(diagnoses)
    
    if not diagnosis_list:
        diagnosis_list = ['Perlu pemeriksaan lebih lanjut']
    
    return diagnosis_list[:3]  # Batasi maksimal 3 diagnosis

def extract_recommendations_from_text(text):
    """Extract rekomendasi dari text"""
    return [
        "Konsultasi dengan dokter umum",
        "Istirahat yang cukup",
        "Konsumsi obat sesuai resep dokter",
        "Monitor perkembangan gejala"
    ]

def fallback_analysis(keluhan, error_msg):
    """Analisis fallback jika AI gagal"""

    keluhan_lower = keluhan.lower()

    # Determine severity and percentage (ubah ke angka fix)
    if any(word in keluhan_lower for word in ['parah', 'sangat sakit', 'tidak bisa', 'emergency']):
        persentase = 85
        urgensi = "tinggi"
    elif any(word in keluhan_lower for word in ['sakit', 'nyeri', 'demam']):
        persentase = 70
        urgensi = "sedang"
    else:
        persentase = 40
        urgensi = "rendah"

    return {
        "persentase_klaim": persentase,
        "kemungkinan_diagnosis": [
            "Perlu pemeriksaan dokter untuk diagnosis yang akurat",
            "Konsultasi dengan spesialis jika diperlukan"
        ],
        "rekomendasi_tindakan": [
            "Segera konsultasi dengan dokter",
            "Bawa riwayat medis lengkap",
            "Catat perkembangan gejala"
        ],
        "tingkat_urgensi": urgensi,
        "dokumen_pendukung": [
            "Surat rujukan dokter",
            "Hasil pemeriksaan laboratorium", 
            "Dokumentasi gejala",
            "Resep obat"
        ],
        "catatan": f"Analisis AI tidak tersedia: {error_msg[:100]}..."
    }