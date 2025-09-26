import os
import logging
from dotenv import load_dotenv
import google.generativeai as genai
from google.oauth2 import service_account

load_dotenv()
GEMINI_CREDENTIAL_JSON = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

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

def get_gemini_model():
    if GEMINI_CREDENTIAL_JSON and GEMINI_CREDENTIAL_JSON.startswith("{"):
        import json
        creds_info = json.loads(GEMINI_CREDENTIAL_JSON)
        credentials = service_account.Credentials.from_service_account_info(creds_info)
    else:
        credentials = service_account.Credentials.from_service_account_file(GEMINI_CREDENTIAL_JSON or "credential.json")
    genai.configure(credentials=credentials)
    return genai.GenerativeModel("gemini-2.5-flash")  # ganti sesuai model kamu

def ask_ai_insurance_guide(user_message):
    global chat_history
    chat_history.append({"role": "user", "content": user_message})

    try:
        prompt = f"{SYSTEM_PROMPT}\n\nRiwayat chat:\n" + "\n".join(
            [f"{msg['role']}: {msg['content']}" for msg in chat_history[-5:]]
        ) + f"\n\nUser: {user_message}\nAI Insurance Guide:"
        model = get_gemini_model()
        response = model.generate_content(prompt)
        assistant_message = response.text.strip()
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