import os
import logging
from dotenv import load_dotenv
import google.generativeai as genai
from google.oauth2 import service_account
import re
import json

load_dotenv()
GEMINI_CREDENTIAL_JSON = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

SYSTEM_PROMPT = """
Anda adalah AI Coverage Assistant untuk asuransi kesehatan.
Tugas Anda:
- Berdasarkan data diagnosis dokter, data asuransi, invoice, dan data lain yang diberikan, isi tabel coverage/tanggungan asuransi berikut:
- Output harus JSON valid dengan field:
  - jenis_layanan
  - deskripsi_layanan
  - status_penanggungan
  - persentasi_penanggungan
  - limit_maksimum
  - sisa_kuota
  - estimasi_biaya_keluar
  - alasan_status
  - tanggal_efektif_penanggungan
  - catatan_tambahan
- Jawab dengan bahasa profesional, jelas, dan ramah.
- Jika ada data yang tidak tersedia, isi dengan string kosong atau penjelasan singkat.
Contoh output:
{
  "jenis_layanan": "Rawat Jalan",
  "deskripsi_layanan": "MRI Otak, CT Scan",
  "status_penanggungan": "Ditanggung",
  "persentasi_penanggungan": "80%",
  "limit_maksimum": "Rp5.000.000 per tahun",
  "sisa_kuota": "Rp1.500.000",
  "estimasi_biaya_keluar": "Rp450.000",
  "alasan_status": "Obat tidak masuk polis",
  "tanggal_efektif_penanggungan": "1 Jan 2025 - 31 Des 2025",
  "catatan_tambahan": "Catatan Tambahan"
}
"""

def get_gemini_model():
    if GEMINI_CREDENTIAL_JSON and GEMINI_CREDENTIAL_JSON.startswith("{"):
        import json as _json
        creds_info = _json.loads(GEMINI_CREDENTIAL_JSON)
        credentials = service_account.Credentials.from_service_account_info(creds_info)
    else:
        credentials = service_account.Credentials.from_service_account_file(GEMINI_CREDENTIAL_JSON or "credential.json")
    genai.configure(credentials=credentials)
    return genai.GenerativeModel("gemini-2.5-flash")  # ganti sesuai model kamu

def coverage_ai_pipeline(diagnosis: dict, asuransi: dict, invoice: dict, extra: dict = None) -> dict:
    prompt = SYSTEM_PROMPT + "\n\nData diagnosis dokter:\n" + str(diagnosis)
    prompt += "\n\nData asuransi:\n" + str(asuransi)
    prompt += "\n\nData invoice:\n" + str(invoice)
    if extra:
        prompt += "\n\nData tambahan:\n" + str(extra)

    try:
        model = get_gemini_model()
        response = model.generate_content(prompt)
        result_text = response.text.strip()
        print("=== AI OUTPUT ===")
        print(result_text)
        # Ambil hanya blok JSON dari output
        match = re.search(r'\{[\s\S]*\}', result_text)
        if match:
            clean_text = match.group(0)
            return json.loads(clean_text)
        else:
            raise ValueError("Tidak ditemukan blok JSON pada output AI")
    except Exception as e:
        logging.error(f"Error in coverage_ai_pipeline: {str(e)}")
        # Fallback jika gagal
        return {
            "jenis_layanan": "",
            "deskripsi_layanan": "",
            "status_penanggungan": "",
            "persentasi_penanggungan": "",
            "limit_maksimum": "",
            "sisa_kuota": "",
            "estimasi_biaya_keluar": "",
            "alasan_status": "",
            "tanggal_efektif_penanggungan": "",
            "catatan_tambahan": "Gagal mengambil data coverage, silakan cek data manual."
        }