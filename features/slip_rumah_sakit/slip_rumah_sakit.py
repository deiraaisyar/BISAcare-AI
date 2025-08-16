from PIL import Image
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os
import tempfile

load_dotenv()
client = InferenceClient(
    model="meta-llama/Llama-3.2-3B-Instruct",
    token=os.getenv("HF_TOKEN")
)

def extract_text(image_bytes):
    """Ekstrak teks dari gambar slip rumah sakit menggunakan OCR."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            temp_file.write(image_bytes)
            temp_file_path = temp_file.name
        image = Image.open(temp_file_path)
        # Gunakan OCR, misal pytesseract (bisa diganti sesuai kebutuhan)
        try:
            import pytesseract
            raw_text = pytesseract.image_to_string(image, lang="ind")
        except ImportError:
            raw_text = ""
        finally:
            os.unlink(temp_file_path)
        return raw_text
    except Exception as e:
        return ""

def parse_slip_with_ai(raw_text):
    """Parse slip rumah sakit dengan AI untuk ekstraksi field penting."""
    prompt = f"""
Berikut adalah hasil OCR dari slip rumah sakit:
{raw_text}

Ekstrak data berikut (isi null jika tidak ditemukan):
- jenis_layanan
- deskripsi_layanan
- status_pertanggungan
- limit_maksimum
- sisa_kuota (rupiah)
- estimasi_biaya_keluar
- alasan_status
- tanggal_efektif_pertanggungan
- catatan_tambahan

Jawab dalam format JSON.
"""
    response = client.text_generation(
        prompt=prompt,
        max_tokens=512,
        temperature=0,
    )
    # Parsing response AI ke dict
    import json
    try:
        result = json.loads(response)
    except Exception:
        result = {
            "jenis_layanan": None,
            "deskripsi_layanan": None,
            "status_pertanggungan": None,
            "limit_maksimum": None,
            "sisa_kuota": None,
            "estimasi_biaya_keluar": None,
            "alasan_status": None,
            "tanggal_efektif_pertanggungan": None,
            "catatan_tambahan": None
        }
    return result