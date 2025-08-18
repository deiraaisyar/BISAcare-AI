import os
import json
from huggingface_hub import InferenceClient

client = InferenceClient(
    model="google/gemini-pro",
    token=os.getenv("GEMINI_API_KEY")
)

def analisis_tanggungan_ai(isi_data, hasil_diagnosis):
    """
    Analisis gabungan data isi_data dan hasil diagnosis dokter menggunakan Gemini.
    """
    prompt = f"""
Data user dari form isi_data:
{json.dumps(isi_data, ensure_ascii=False, indent=2)}

Hasil diagnosis dokter:
{json.dumps(hasil_diagnosis, ensure_ascii=False, indent=2)}

Analisis gabungan: Berikan saran, kemungkinan diagnosis, dan langkah selanjutnya untuk proses asuransi. Jawab singkat, jelas, dan profesional.
"""
    messages = [{"role": "user", "content": prompt}]
    response = client.chat_completion(
        messages=messages,
        max_tokens=512,
        temperature=0.2,
        stream=False
    )
    ai_result = response.choices[0].message.content
    return {
        "status": "success",
        "analisis_tanggungan": ai_result,
        "isi_data": isi_data,
        "hasil_diagnosis": hasil_diagnosis
    }