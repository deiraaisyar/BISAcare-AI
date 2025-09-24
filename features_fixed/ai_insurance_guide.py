import os
import logging
import requests
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

SYSTEM_PROMPT = """
Anda adalah AI Insurance Guide, asisten AI yang membantu pengguna membuat dan memahami proses pembuatan asuransi.

INSTRUKSI:
- Pandu user langkah demi langkah membuat asuransi (misal: dokumen yang dibutuhkan, cara memilih produk, proses pendaftaran, dsb)
- Jawab dengan bahasa ramah, profesional, dan mudah dipahami
- Jika user bertanya tentang produk, berikan penjelasan singkat dan saran sesuai kebutuhan user
- Jika informasi tidak lengkap, sarankan untuk menghubungi customer service atau agen asuransi
- Jangan memberikan saran medis, hanya panduan administratif dan produk asuransi
"""

chat_history = []

def ask_ai_insurance_guide(user_message):
    global chat_history
    chat_history.append({"role": "user", "content": user_message})

    try:
        prompt = f"{SYSTEM_PROMPT}\n\nRiwayat chat:\n" + "\n".join(
            [f"{msg['role']}: {msg['content']}" for msg in chat_history[-5:]]
        ) + f"\n\nUser: {user_message}\nAI Insurance Guide:"

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
        assistant_message = result_text.strip()

        chat_history.append({"role": "assistant", "content": assistant_message})
        if len(chat_history) > 20:
            chat_history = chat_history[-20:]
        return assistant_message

    except Exception as e:
        logging.error(f"Error in ask_ai_insurance_guide: {str(e)}")
        fallback_message = (
            "Maaf, AI Insurance Guide sedang mengalami kendala teknis. "
            "Silakan hubungi customer service perusahaan asuransi atau agen asuransi untuk bantuan pembuatan asuransi."
        )
        chat_history.append({"role": "assistant", "content": fallback_message})
        return fallback_message

def get_ai_insurance_guide_history():
    return chat_history

def clear_ai_insurance_guide_history():
    global chat_history
    chat_history = []
    logging.info("AI Insurance Guide chat history cleared")