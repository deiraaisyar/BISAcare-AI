import os
import logging
import requests
from dotenv import load_dotenv

try:
    from rag.retriever import SimpleRAGRetriever
    RAG_AVAILABLE = True
except ImportError as e:
    RAG_AVAILABLE = False
    logging.warning(f"RAG components not available: {str(e)}")

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

rag_retriever = None
rag_initialized = False

def initialize_rag():
    global rag_retriever, rag_initialized
    if rag_initialized:
        return
    rag_initialized = True
    if not RAG_AVAILABLE:
        logging.warning("RAG not available - continuing without document retrieval")
        return
    try:
        rag_retriever = SimpleRAGRetriever()
        if rag_retriever.is_available():
            logging.info("RAG retriever initialized successfully")
        else:
            logging.warning("RAG retriever initialized but no documents available")
            rag_retriever = None
    except Exception as e:
        logging.error(f"Failed to initialize RAG retriever: {str(e)}")
        rag_retriever = None

SYSTEM_PROMPT = """Anda adalah BISAbot, asisten AI yang membantu pengguna memahami produk asuransi.

INSTRUKSI PENTING:
- Jika tersedia informasi dari dokumen asuransi, gunakan sebagai referensi utama, tidak usah dicantumkan sumber dokumennya, asalkan sesuai
- Berikan jawaban yang akurat, jelas, dan mudah dipahami  
- Jika informasi tidak lengkap, berikan jawaban sesuai pengetahuan anda dan memberikan saran untuk menghubungi customer service di akhir jawaban
- Gunakan bahasa yang ramah dan profesional
- Fokus pada memberikan informasi yang bermanfaat tentang asuransi

Jika Anda tidak yakin tentang informasi tertentu, katakan dengan jujur dan sarankan untuk verifikasi lebih lanjut.
"""

chat_history = []

def ask_bisabot(user_message):
    global chat_history
    initialize_rag()
    chat_history.append({"role": "user", "content": user_message})

    try:
        context = ""
        if rag_retriever and rag_retriever.is_available():
            context = rag_retriever.get_context_for_query(user_message)
        # Prompt selalu gabungkan context RAG (jika ada) dan instruksi umum
        if context.strip():
            prompt = f"{SYSTEM_PROMPT}\n\nBerikut adalah informasi dari dokumen asuransi yang relevan:\n{context}\n\n---\nPertanyaan pengguna: {user_message}\nBerikan jawaban berdasarkan informasi di atas. Jika informasi tidak lengkap, tambahkan saran untuk menghubungi customer service."
        else:
            prompt = f"{SYSTEM_PROMPT}\n\nPertanyaan pengguna: {user_message}\nBerikan jawaban umum tentang asuransi berdasarkan pengetahuan Anda dan sarankan untuk menghubungi customer service untuk informasi detail dan terkini."

        # Ganti ke Gemini 1.5 Flash seperti di test_gemini.py
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
        logging.error(f"Error in ask_bisabot: {str(e)}")
        fallback_message = """Maaf, saya mengalami kendala teknis saat ini. Untuk mendapatkan informasi asuransi yang akurat dan terkini, silakan:
1. Hubungi customer service perusahaan asuransi
2. Kunjungi website resmi perusahaan asuransi
3. Konsultasi dengan agen asuransi terdekat

Terima kasih atas pengertian Anda."""
        chat_history.append({"role": "assistant", "content": fallback_message})
        return fallback_message

def get_chat_history():
    return chat_history

def clear_chat_history():
    global chat_history
    chat_history = []
    logging.info("Chat history cleared")
