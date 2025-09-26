import os
import logging
import re
import json
from dotenv import load_dotenv
import google.generativeai as genai
from google.oauth2 import service_account

load_dotenv()
GEMINI_CREDENTIAL_JSON = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

SYSTEM_PROMPT = """
Anda adalah Asisten Rumah Sakit bagian Asuransi.
Tugas Anda:
- Membantu menilai keluhan pasien terkait klaim asuransi.
- Output WAJIB berupa JSON valid, tanpa penjelasan tambahan, tanpa kata 'Health Anda' di setiap poin.
- Berikan output berupa:
  1. persentase_kondisi_dapat_diklaim (angka dalam %)
  2. kemungkinan_diagnosis (poin-poin)
  3. rekomendasi_tindakan (poin-poin)
  4. dokumen_pendukung_klaim (poin-poin)
- Output hanya JSON, tanpa kata lain.
Contoh output:
{
  "persentase_kondisi_dapat_diklaim": 85,
  "kemungkinan_diagnosis": ["Penyakit jantung koroner", "Hipertensi"],
  "rekomendasi_tindakan": ["Segera periksa ke rumah sakit", "Simpan rekam medis"],
  "dokumen_pendukung_klaim": ["Hasil pemeriksaan jantung", "Surat keterangan dokter"]
}
"""

FALLBACK = {
    "persentase_kondisi_dapat_diklaim": 0,
    "kemungkinan_diagnosis": ["Tidak dapat menentukan."],
    "rekomendasi_tindakan": ["Hubungi pihak rumah sakit atau customer service."],
    "dokumen_pendukung_klaim": ["Konsultasikan ke administrasi rumah sakit."]
}

def get_gemini_model():
    if GEMINI_CREDENTIAL_JSON and GEMINI_CREDENTIAL_JSON.startswith("{"):
        import json
        creds_info = json.loads(GEMINI_CREDENTIAL_JSON)
        credentials = service_account.Credentials.from_service_account_info(creds_info)
    else:
        credentials = service_account.Credentials.from_service_account_file(GEMINI_CREDENTIAL_JSON or "credential.json")
    genai.configure(credentials=credentials)
    return genai.GenerativeModel("gemini-2.5-flash")  

def process_keluhan(text_keluhan: str):
    prompt = f"{SYSTEM_PROMPT}\n\nKeluhan pasien:\n{text_keluhan}"

    try:
        model = get_gemini_model()
        response = model.generate_content(prompt)
        result_text = response.text.strip()

        print("=== AI TEXT ===")
        print(result_text)
        
        clean_text = re.sub(r'```json', '', result_text, flags=re.IGNORECASE)
        clean_text = re.sub(r'```', '', clean_text)
        clean_text = clean_text.strip()

        try:
            result_json = json.loads(clean_text)
        except json.JSONDecodeError:
            logging.warning("Output AI masih tidak valid JSON, gunakan parse_text_response")
            result_json = parse_text_response(result_text)

        for key in FALLBACK.keys():
            if key not in result_json:
                result_json[key] = FALLBACK[key]

        print("=== PARSED JSON ===")
        print(result_json)

        return result_json

    except Exception as e:
        logging.error(f"Error in process_keluhan: {str(e)}")
        return FALLBACK

def parse_text_response(text: str):
    persentase = re.search(r'(\d+)%', text)
    persentase = int(persentase.group(1)) if persentase else 0
    bullets = [b for b in re.findall(r'[-•]\s*(.+)', text) if "Health Anda" not in b]
    diagnosis = bullets[0:2] if len(bullets) >= 2 else ["Tidak dapat menentukan."]
    tindakan = bullets[2:4] if len(bullets) >= 4 else ["Hubungi pihak rumah sakit atau customer service."]
    dokumen = bullets[4:6] if len(bullets) >= 6 else ["Konsultasikan ke administrasi rumah sakit."]
    return {
        "persentase_kondisi_dapat_diklaim": persentase,
        "kemungkinan_diagnosis": diagnosis,
        "rekomendasi_tindakan": tindakan,
        "dokumen_pendukung_klaim": dokumen
    }
