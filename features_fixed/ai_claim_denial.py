import os
import requests
import logging
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

SYSTEM_PROMPT = """
Anda adalah AI Claim Denial Rewriter, asisten AI yang membantu user memahami alasan penolakan klaim asuransi dan memberi saran langkah selanjutnya.

INSTRUKSI:
- Cek data yang diberikan user (berdasarkan schemas.py).
- Jika ada data yang kurang, informasikan dengan bahasa ramah dan jelas, serta beri saran apa yang harus dilengkapi.
- Jika klaim ditolak, berikan alasan penolakan dan langkah-langkah yang bisa dilakukan user untuk banding atau melengkapi dokumen.
- Jawab dalam format chatbot, gunakan sapaan dan instruksi singkat.
"""

def check_missing_fields(data: dict, required_fields: list) -> list:
    missing = []
    for field in required_fields:
        value = data.get(field)
        if value is None or (isinstance(value, str) and not value.strip()):
            missing.append(field)
    return missing

def claim_denial_chatbot(user_message: str, claim_data: dict, required_fields: list):
    missing_fields = check_missing_fields(claim_data, required_fields)
    if missing_fields:
        missing_str = ", ".join(missing_fields)
        feedback = (
            f"Selamat Siang! Berikut adalah beberapa hal yang harus kamu lengkapi:\n"
            f"Kamu belum mengisi/mengupload data berikut: {missing_str}.\n"
            f"Silakan lengkapi data tersebut agar proses klaim bisa dilanjutkan."
        )
        return feedback

    # Jika tidak ada data kurang, gunakan Gemini untuk analisis penolakan dan saran
    prompt = (
        SYSTEM_PROMPT +
        "\n\nData klaim user:\n" + str(claim_data) +
        "\n\nPesan user:\n" + user_message +
        "\n\nBerikan alasan penolakan (jika ada) dan saran langkah selanjutnya."
    )

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
        return result_text
    except Exception as e:
        logging.error(f"Error in claim_denial_chatbot: {str(e)}")
        return (
            "Maaf, AI Claim Denial Rewriter sedang mengalami kendala teknis. "
            "Silakan hubungi customer service asuransi untuk bantuan lebih lanjut."
        )