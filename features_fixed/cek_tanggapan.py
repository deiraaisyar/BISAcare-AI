import os
import logging
import requests
import re
import json
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

SYSTEM_PROMPT = """
Anda adalah Asisten Rumah Sakit bagian Asuransi.
Tugas Anda:
- Membantu menilai keluhan pasien terkait klaim asuransi.
- Berikan output berupa:
  1. persentase_kondisi_dapat_diklaim (angka dalam %)
  2. kemungkinan_diagnosis (poin-poin)
  3. rekomendasi_tindakan (poin-poin)
  4. dokumen_pendukung_klaim (poin-poin)
- Gunakan bahasa profesional, jelas, dan ramah.
- Buat poin-poin yang mudah dibaca dan dipahami.
- Jika ada ketidakpastian, berikan catatan agar pasien memverifikasi ke pihak rumah sakit.
- Output harus JSON valid.
Contoh output yang benar:
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

def process_keluhan(text_keluhan: str):
    prompt = f"{SYSTEM_PROMPT}\n\nKeluhan pasien:\n{text_keluhan}"

    try:
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"
        headers = {"Content-Type": "application/json"}
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        params = {"key": GEMINI_API_KEY}

        response = requests.post(url, headers=headers, params=params, json=payload, timeout=30)
        response.raise_for_status()
        result_json_raw = response.json()

        print("=== RAW GEMINI RESPONSE ===")
        print(result_json_raw)

        result_text = result_json_raw["candidates"][0]["content"]["parts"][0]["text"].strip()

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
    """
    Ambil persentase + bullet points dari teks bebas AI
    """
    persentase = re.search(r'(\d+)%', text)
    persentase = int(persentase.group(1)) if persentase else 0
    bullets = re.findall(r'[-•]\s*(.+)', text)
    diagnosis = bullets[0:2] if len(bullets) >= 2 else ["Tidak dapat menentukan."]
    tindakan = bullets[2:4] if len(bullets) >= 4 else ["Hubungi pihak rumah sakit atau customer service."]
    dokumen = bullets[4:6] if len(bullets) >= 6 else ["Konsultasikan ke administrasi rumah sakit."]

    return {
        "persentase_kondisi_dapat_diklaim": persentase,
        "kemungkinan_diagnosis": diagnosis,
        "rekomendasi_tindakan": tindakan,
        "dokumen_pendukung_klaim": dokumen
    }
