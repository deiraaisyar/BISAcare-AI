import pytesseract
from PIL import Image
import io
import os
import requests
from dotenv import load_dotenv
import pytesseract
from PIL import Image
import io

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def extract_text(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    text = pytesseract.image_to_string(image, lang="ind")
    return text

def extract_slip_text(image_bytes):
    """
    Ekstrak teks dari foto slip rumah sakit menggunakan OCR.
    """
    image = Image.open(io.BytesIO(image_bytes))
    text = pytesseract.image_to_string(image, lang="ind")
    return text

def parse_with_ai(text):
    prompt = f"""
Dari teks hasil OCR berikut:
{text}

Pisahkan menjadi 2 bagian: informasi KTP dan informasi Polis Asuransi.
Formatkan ke JSON dengan field:
- KTP: nama, kelurahan_desa, kecamatan, nama_provinsi, nama_daerah
- Polis: nama_asuransi
Jawab hanya JSON saja.
"""
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ]
    }
    params = {"key": GEMINI_API_KEY}
    response = requests.post(url, headers=headers, params=params, json=payload, timeout=30)
    response.raise_for_status()
    result_text = response.json()["candidates"][0]["content"]["parts"][0]["text"]