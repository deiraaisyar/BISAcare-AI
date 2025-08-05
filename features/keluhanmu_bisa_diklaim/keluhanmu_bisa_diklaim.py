import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import json
import re
import whisper
import tempfile
import shutil
import logging

# Setup logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Initialize Hugging Face client
client = InferenceClient(
    model="meta-llama/Llama-3.2-3B-Instruct",
    token=os.getenv("HF_TOKEN")
)

# Load Whisper model (using base model for balance of speed and accuracy)
whisper_model = None

def load_whisper_model():
    """Load Whisper model lazily"""
    global whisper_model
    if whisper_model is None:
        logger.info("Loading Whisper model...")
        try:
            whisper_model = whisper.load_model("base")
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {str(e)}")
            raise e
    return whisper_model

MEDICAL_ANALYSIS_PROMPT = """Anda adalah asisten AI medis yang membantu menganalisis keluhan kesehatan untuk asuransi.
Berdasarkan keluhan yang diberikan, berikan analisis dalam format JSON:

{
    "persentase_klaim": "angka 0-100",
    "kemungkinan_diagnosis": "daftar kemungkinan diagnosis",
    "rekomendasi_tindakan": "saran tindakan medis",
    "tingkat_urgensi": "rendah/sedang/tinggi",
    "dokumen_pendukung": "daftar dokumen yang diperlukan"
}

Berikan jawaban yang akurat dan profesional."""

def transcribe_audio(audio_file_path):
    """
    Convert audio file to text using OpenAI Whisper
    Supports MP4, MP3, WAV, M4A, FLAC, OGG, WebM
    """
    try:
        logger.info(f"Starting transcription for file: {audio_file_path}")
        
        # Check if file exists and has content
        if not os.path.exists(audio_file_path):
            raise Exception(f"Audio file not found: {audio_file_path}")
        
        file_size = os.path.getsize(audio_file_path)
        logger.info(f"Audio file size: {file_size} bytes")
        
        if file_size == 0:
            raise Exception("Audio file is empty")
        
        model = load_whisper_model()
        
        # Try multiple approaches for better transcription
        logger.info("Starting Whisper transcription...")
        
        # Approach 1: Auto-detect language first
        try:
            result = model.transcribe(
                audio_file_path,
                language=None,  # Auto-detect language
                fp16=False,
                verbose=True  # Enable verbose for debugging
            )
            logger.info(f"Auto-detect result: {result}")
        except Exception as e:
            logger.warning(f"Auto-detect failed: {str(e)}, trying with Indonesian language...")
            
            # Approach 2: Force Indonesian language
            result = model.transcribe(
                audio_file_path,
                language='id',  # Force Indonesian
                fp16=False,
                verbose=True,
                task='transcribe'  # Explicitly set task
            )
            logger.info(f"Indonesian language result: {result}")
        
        transcribed_text = result.get("text", "").strip()
        detected_language = result.get("language", "unknown")
        
        logger.info(f"Transcribed text: '{transcribed_text}'")
        logger.info(f"Detected language: {detected_language}")
        
        if not transcribed_text:
            # Try one more time with different settings
            logger.warning("Empty transcription, trying with 'tiny' model...")
            tiny_model = whisper.load_model("tiny")
            result = tiny_model.transcribe(
                audio_file_path,
                language='id',
                fp16=False
            )
            transcribed_text = result.get("text", "").strip()
            logger.info(f"Tiny model result: '{transcribed_text}'")
        
        if not transcribed_text:
            raise Exception("Transcription resulted in empty text. Audio might be silent or corrupt.")
        
        return transcribed_text
        
    except Exception as e:
        logger.error(f"Error in transcribe_audio: {str(e)}")
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
    """Parse response text menjadi format yang diinginkan"""
    
    # Analisis sederhana berdasarkan keyword
    serious_symptoms = ['nyeri dada', 'sesak napas', 'demam tinggi', 'muntah darah', 'pingsan', 'stroke']
    moderate_symptoms = ['demam', 'batuk', 'sakit kepala', 'mual', 'diare', 'nyeri perut']
    mild_symptoms = ['flu', 'pilek', 'batuk ringan', 'pusing ringan']
    
    keluhan_lower = keluhan.lower()
    
    # Tentukan persentase berdasarkan severity
    if any(symptom in keluhan_lower for symptom in serious_symptoms):
        persentase = "85-95"
        urgensi = "tinggi"
    elif any(symptom in keluhan_lower for symptom in moderate_symptoms):
        persentase = "60-80"
        urgensi = "sedang"
    else:
        persentase = "30-60"
        urgensi = "rendah"
    
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
    
    # Analisis keyword sederhana
    keluhan_lower = keluhan.lower()
    
    # Determine severity and percentage
    if any(word in keluhan_lower for word in ['parah', 'sangat sakit', 'tidak bisa', 'emergency']):
        persentase = "80-90"
        urgensi = "tinggi"
    elif any(word in keluhan_lower for word in ['sakit', 'nyeri', 'demam']):
        persentase = "60-75"
        urgensi = "sedang"  
    else:
        persentase = "40-60"
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

# Test function
if __name__ == "__main__":
    test_keluhan = "Saya mengalami demam tinggi 39 derajat, batuk berdahak, dan sesak napas sejak 3 hari yang lalu"
    result = analyze_health_complaint(test_keluhan)
    print(json.dumps(result, indent=2, ensure_ascii=False))