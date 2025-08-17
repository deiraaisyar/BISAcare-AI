from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os
from features.bisabot.bisabot import get_chat_history

load_dotenv()
client = InferenceClient(
    model="meta-llama/Llama-3.2-3B-Instruct",
    token=os.getenv("HF_TOKEN")
)

def cek_data_isi_data(data_isi):
    """
    Mengecek data hasil isi_data dan memberi saran jika ada field yang masih kosong/null.
    Menggabungkan insight dari chat history BISAbot dan AI reasoning.
    """
    saran = []
    ktp = data_isi.get("ktp", {})
    if not ktp or any(v in [None, "", "null"] for v in ktp.values()):
        saran.append("Lengkapi data KTP. Jika belum punya, silakan unggah foto KTP yang jelas.")

    polis = data_isi.get("polis", {})
    if not polis or any(v in [None, "", "null"] for v in polis.values()):
        saran.append("Lengkapi data Polis Asuransi. Jika belum punya, minta salinan polis ke rumah sakit atau asuransi.")
        
    for field, label in [
        ("nomor_polis", "Nomor polis asuransi"),
        ("layanan", "Jenis layanan"),
        ("nomor_hp", "Nomor HP aktif"),
        ("keluhan", "Keluhan kesehatan"),
    ]:
        if data_isi.get(field) in [None, "", "null"]:
            saran.append(f"Isi {label} pada form.")

    history = get_chat_history()
    last_message = history[-1]["message"] if history else ""

    prompt = f"""
Data user:
{data_isi}

Riwayat chat BISAbot terakhir:
{last_message}

Berdasarkan data di atas, berikan saran langkah selanjutnya yang harus dilakukan user agar proses klaim asuransi bisa berjalan lancar. Jika ada data yang kurang, beri tahu dokumen/form apa yang perlu dilengkapi. Jawab singkat dan jelas.
"""
    messages = [{"role": "user", "content": prompt}]
    ai_response = client.chat_completion(
        messages=messages,
        max_tokens=256,
        temperature=0,
        stream=False
    ).choices[0].message.content

    saran.append(f"{ai_response}")

    if not saran:
        saran.append("Data sudah lengkap. Kamu bisa melanjutkan proses klaim atau konsultasi lebih lanjut.")

    return {
        "status": "cek_data",
        "saran": saran,
        "data_isi": data_isi,
        "chat_history": history
    }