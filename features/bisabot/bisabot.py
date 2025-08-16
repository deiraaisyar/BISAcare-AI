import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import logging

# Import RAG components with fallback
try:
    from rag.retriever import SimpleRAGRetriever
    RAG_AVAILABLE = True
except ImportError as e:
    RAG_AVAILABLE = False
    logging.warning(f"RAG components not available: {str(e)}")

load_dotenv()

# Initialize Hugging Face client
client = InferenceClient(
    model="meta-llama/Llama-3.2-3B-Instruct",
    token=os.getenv("HF_TOKEN")
)

# Initialize RAG retriever (lazy loading)
rag_retriever = None
rag_initialized = False

def initialize_rag():
    """Initialize RAG system (lazy loading)"""
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

# Global variable to store chat history
chat_history = []

def ask_bisabot(user_message):
    """Main function to chat with BISAbot (with RAG integration)"""
    global chat_history
    
    # Initialize RAG if not done yet
    initialize_rag()
    
    # Add user message to history
    chat_history.append({"role": "user", "content": user_message})
    
    try:
        # Get relevant context from RAG if available
        context = ""
        if rag_retriever and rag_retriever.is_available():
            context = rag_retriever.get_context_for_query(user_message)
        
        # Prepare enhanced prompt with context
        if context.strip():
            enhanced_message = f"""
Berikut adalah informasi dari dokumen asuransi yang relevan:

{context}

---

Pertanyaan pengguna: {user_message}

Berikan jawaban berdasarkan informasi di atas. Jika informasi tidak lengkap, tambahkan saran untuk menghubungi customer service.
"""
        else:
            enhanced_message = f"""
Pertanyaan pengguna: {user_message}

Berikan jawaban umum tentang asuransi berdasarkan pengetahuan Anda dan sarankan untuk menghubungi customer service untuk informasi detail dan terkini.
"""
        
        # Prepare messages for the model
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *chat_history[:-1],  # Previous chat history (exclude current user message)
            {"role": "user", "content": enhanced_message}
        ]
        
        # Get response from AI
        response = client.chat_completion(
            messages=messages,
            max_tokens=512,
            temperature=0.2,
            stream=False
        )
        
        assistant_message = response.choices[0].message.content
        
        # Add assistant response to history (with original user message, not enhanced)
        chat_history.append({"role": "assistant", "content": assistant_message})
        
        # Keep only last 10 exchanges to prevent context overflow
        if len(chat_history) > 20:  # 20 = 10 user + 10 assistant messages
            chat_history = chat_history[-20:]
        
        return assistant_message
        
    except Exception as e:
        logging.error(f"Error in ask_bisabot: {str(e)}")
        
        # Fallback responses based on keywords if AI fails
        fallback_message = get_fallback_response(user_message)
        
        # Add fallback to history
        chat_history.append({"role": "assistant", "content": fallback_message})
        
        return fallback_message

def get_fallback_response(user_message: str) -> str:
    """Generate fallback response based on keywords"""
    user_message_lower = user_message.lower()
    
    if any(keyword in user_message_lower for keyword in ['premi', 'biaya', 'harga', 'tarif']):
        return """Untuk informasi premi dan biaya asuransi yang akurat, saya sarankan Anda untuk:
1. Menghubungi customer service perusahaan asuransi
2. Mengunjungi website resmi perusahaan asuransi  
3. Berkonsultasi dengan agen asuransi
Premi dapat bervariasi tergantung usia, kondisi kesehatan, dan jenis perlindungan yang dipilih."""
        
    elif any(keyword in user_message_lower for keyword in ['klaim', 'reimburse', 'penggantian', 'ganti rugi']):
        return """Untuk proses klaim asuransi, langkah umumnya adalah:
1. Segera laporkan kejadian kepada perusahaan asuransi
2. Siapkan dokumen lengkap (formulir klaim, kwitansi, hasil pemeriksaan)
3. Submit dokumen sesuai prosedur yang ditetapkan
4. Tunggu proses verifikasi dan persetujuan

Untuk detail prosedur yang spesifik, silakan hubungi customer service."""
        
    elif any(keyword in user_message_lower for keyword in ['manfaat', 'benefit', 'coverage', 'pertanggungan']):
        return """Manfaat asuransi kesehatan umumnya meliputi:
1. Rawat inap di rumah sakit
2. Rawat jalan dan konsultasi dokter
3. Obat-obatan dan pemeriksaan laboratorium
4. Tindakan medis dan operasi

Detail manfaat tergantung pada jenis polis yang Anda pilih. Untuk informasi lengkap manfaat polis Anda, silakan cek polis atau hubungi customer service."""
        
    elif any(keyword in user_message_lower for keyword in ['daftar', 'beli', 'pembelian', 'bergabung']):
        return """Untuk mendaftar asuransi, Anda bisa:
1. Menghubungi agen asuransi
2. Mengunjungi kantor cabang perusahaan asuransi
3. Mendaftar online melalui website resmi
4. Menghubungi customer service untuk konsultasi

Siapkan dokumen identitas dan informasi kesehatan yang diperlukan."""
        
    else:
        return """Maaf, saya mengalami kendala teknis saat ini. Untuk mendapatkan informasi asuransi yang akurat dan terkini, silakan:
1. Hubungi customer service perusahaan asuransi
2. Kunjungi website resmi perusahaan asuransi
3. Konsultasi dengan agen asuransi terdekat

Terima kasih atas pengertian Anda."""

def get_chat_history():
    """Get current chat history"""
    return chat_history

def clear_chat_history():
    """Clear chat history"""
    global chat_history
    chat_history = []
    logging.info("Chat history cleared")

def get_rag_status():
    """Get RAG system status"""
    if not rag_retriever:
        return {"available": False, "reason": "RAG not initialized"}
    elif not rag_retriever.is_available():
        return {"available": False, "reason": "No documents loaded"}
    else:
        return {
            "available": True, 
            "documents_count": len(rag_retriever.documents),
            "index_ready": rag_retriever.index is not None
        }

# # Test function
# if __name__ == "__main__":
#     print("Testing BISAbot with RAG...")
    
#     # Test RAG status
#     status = get_rag_status()
#     print(f"RAG Status: {status}")
    
#     # Test questions
#     test_questions = [
#         "Apa itu asuransi kesehatan?",
#         "Bagaimana cara klaim asuransi?", 
#         "Berapa premi asuransi kesehatan?",
#         "Apa saja manfaat yang ditanggung?"
#     ]
    
#     for question in test_questions:
#         print(f"\nQ: {question}")
#         answer = ask_bisabot(question)
#         print(f"A: {answer}")
#         print("-" * 80)