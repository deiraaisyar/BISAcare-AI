import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()

# Initialize Hugging Face client
client = InferenceClient(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    token=os.getenv("HF_TOKEN")
)

SYSTEM_PROMPT = """Anda adalah BISAbot, chatbot AI yang membantu pengguna memahami asuransi kesehatan.
Jawaban Anda harus ramah, jelas, dan hanya berdasarkan informasi umum asuransi.
Jika Anda tidak yakin, jawab: "Maaf, saya tidak memiliki informasi itu saat ini." """

# Global variable to store chat history
chat_history = []

def ask_bisabot(user_message):
    global chat_history
    
    # Add user message to history
    chat_history.append({"role": "user", "content": user_message})
    
    # Prepare messages for the model
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + chat_history
    
    try:
        response = client.chat_completion(
            messages=messages,
            max_tokens=512,
            temperature=0.2,
            stream=False
        )
        
        assistant_message = response.choices[0].message.content
        
        # Add assistant response to history
        chat_history.append({"role": "assistant", "content": assistant_message})
        
        # Keep only last 10 exchanges to prevent context overflow
        if len(chat_history) > 20:  # 20 = 10 user + 10 assistant messages
            chat_history = chat_history[-20:]
        
        return assistant_message
        
    except Exception as e:
        error_message = f"Maaf, terjadi kesalahan: {str(e)}"
        chat_history.append({"role": "assistant", "content": error_message})
        return error_message

def get_chat_history():
    """Return the current chat history"""
    return chat_history

def clear_chat_history():
    """Clear the chat history"""
    global chat_history
    chat_history = [] 
