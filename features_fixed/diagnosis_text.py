import os
import requests
import logging
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

SYSTEM_PROMPT = """
Anda adalah AI Diagnosis Extractor. 
Tugas Anda: Ekstrak informasi diagnosis medis dari teks yang diberikan user.
Output harus JSON valid dengan field:
- diagnosa (list of string)
- jenis_kelamin (string)
- nama_dokter (string)
- nama_pasien (string)
- nik (string)
- penyakit (string)
- raw (string, isi dengan teks asli user)
Jika data tidak tersedia, isi dengan string kosong atau list kosong.
"""

def diagnosis_text_pipeline(diagnosis_text: str) -> dict:
    prompt = SYSTEM_PROMPT + "\n\nTeks diagnosis:\n" + diagnosis_text
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ]
    }
    params = {"key": GEMINI_API_KEY}
    try:
        response = requests.post(url, headers=headers, params=params, json=payload, timeout=30)
        response.raise_for_status()
        result_text = response.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        import re, json
        clean_text = re.sub(r'```json', '', result_text, flags=re.IGNORECASE)
        clean_text = re.sub(r'```', '', clean_text).strip()
        result = json.loads(clean_text)
        # Pastikan raw diisi dengan diagnosis_text
        result["raw"] = diagnosis_text
        return result
    except Exception as e:
        logging.error(f"Error in diagnosis_text_pipeline: {str(e)}")
        return {
            "diagnosa": [],
            "jenis_kelamin": "",
            "nama_dokter": "",
            "nama_pasien": "",
            "nik": "",
            "penyakit": "",
            "raw": diagnosis_text
        }