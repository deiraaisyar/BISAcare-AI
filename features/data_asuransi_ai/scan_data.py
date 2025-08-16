import pytesseract
from PIL import Image
import io
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

load_dotenv()
client = InferenceClient(
    model="meta-llama/Llama-3.2-3B-Instruct",
    token=os.getenv("HF_TOKEN")
)

def extract_text(image_bytes):
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
- Polis: jenis_layanan, nama_asuransi
Jawab hanya JSON saja.
"""
    messages = [{"role": "user", "content": prompt}]
    response = client.chat_completion(
        messages=messages,
        max_tokens=512,
        temperature=0,
        stream=False
    )
    return response.choices[0].message.content