import os
import logging
from dotenv import load_dotenv
import google.generativeai as genai
from google.oauth2 import service_account
import re
import json

load_dotenv()
GEMINI_CREDENTIAL_JSON = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

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

def get_gemini_model():
    if GEMINI_CREDENTIAL_JSON and GEMINI_CREDENTIAL_JSON.startswith("{"):
        import json as _json
        creds_info = _json.loads(GEMINI_CREDENTIAL_JSON)
        credentials = service_account.Credentials.from_service_account_info(creds_info)
    else:
        credentials = service_account.Credentials.from_service_account_file(GEMINI_CREDENTIAL_JSON or "credential.json")
    genai.configure(credentials=credentials)
    return genai.GenerativeModel("gemini-2.5-flash")  # ganti sesuai model kamu

def diagnosis_text_pipeline(diagnosis_text: str) -> dict:
    prompt = SYSTEM_PROMPT + "\n\nTeks diagnosis:\n" + diagnosis_text
    try:
        model = get_gemini_model()
        response = model.generate_content(prompt)
        result_text = response.text.strip()
        clean_text = re.sub(r'```json', '', result_text, flags=re.IGNORECASE)
        clean_text = re.sub(r'```', '', clean_text).strip()
        result = json.loads(clean_text)
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