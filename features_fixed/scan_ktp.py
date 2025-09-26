import base64
import os, json
import re
from google.cloud import documentai_v1 as documentai
from google.oauth2 import service_account

PROJECT_ID = "1081333106174"
LOCATION = "us"
PROCESSOR_ID = "d788d904b365af4"

def get_docai_client():
    credentials = None

    gac = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

    if gac and gac.startswith("{"):
        # Cloud Run secret sebagai ENV VAR JSON string
        creds_info = json.loads(gac)
        credentials = service_account.Credentials.from_service_account_info(creds_info)
    elif gac:
        # Lokal atau Cloud Run secret file
        credentials = service_account.Credentials.from_service_account_file(gac)
    else:
        # fallback lokal file
        credentials = service_account.Credentials.from_service_account_file("credential.json")

    return documentai.DocumentProcessorServiceClient(credentials=credentials)


def fallback_parse_raw(raw_text, field_name):
    # Contoh sederhana, bisa disesuaikan dengan format raw
    patterns = {
        "nik": r"NIK\s*[:\-]?\s*(\d+)",
        "nama": r"Nama\s*[:\-]?\s*([A-Z ]+)",
        "tempat_lahir": r"Tempat/Tgl Lahir\s*[:\-]?\s*([A-Z]+)",
        "tanggal_lahir": r"Tempat/Tgl Lahir\s*[:\-]?\s*[A-Z]+,\s*([\d\-]+)",
        "jenis_kelamin": r"Jenis kelamin\s*[:\-]?\s*([A-Z]+)",
        "agama": r"Agama\s*[:\-]?\s*([A-Z]+)",
        "status_perkawinan": r"Status Perkawinan\s*[:\-]?\s*([A-Z ]+)",
        "pekerjaan": r"Pekerjaan\s*[:\-]?\s*([A-Z\/ ]+)",
        "kewarganegaraan": r"Kewarganegaraan\s*[:\-]?\s*([A-Z]+)",
        "kabupaten": r"KABUPATEN\s*[:\-]?\s*([A-Z ]+)",
        "provinsi": r"PROVINSI\s*[:\-]?\s*([A-Z ]+)",
        "kelurahan_desa": r"Kel/Desa\s*[:\-]?\s*([A-Z ]+)",
        "kecamatan": r"Kecamatan\s*[:\-]?\s*([A-Z ]+)",
    }
    pattern = patterns.get(field_name)
    if pattern:
        match = re.search(pattern, raw_text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None

def scan_ktp_pipeline(image_base64: str) -> dict:
    """Pipeline OCR KTP menggunakan Google Document AI"""
    client = get_docai_client()
    image_bytes = base64.b64decode(image_base64)

    # Build resource name
    name = f"projects/{PROJECT_ID}/locations/{LOCATION}/processors/{PROCESSOR_ID}"

    # Build request
    raw_document = documentai.RawDocument(
        content=image_bytes,
        mime_type="image/jpeg"
    )
    request = documentai.ProcessRequest(
        name=name,
        raw_document=raw_document
    )

    result = client.process_document(request=request)
    doc = result.document

    def get_field(field_name):
        for entity in doc.entities:
            if field_name.lower() in entity.type_.lower():
                return entity.mention_text
        # Fallback ke raw jika entity tidak ditemukan
        return fallback_parse_raw(doc.text, field_name)

    parsed = {
        "nik": get_field("nik"),
        "nama": get_field("nama"),
        "tempat_lahir": get_field("tempat_lahir"),
        "tanggal_lahir": get_field("tanggal_lahir"),
        "jenis_kelamin": get_field("jenis_kelamin"),
        "agama": get_field("agama"),
        "status_perkawinan": get_field("status_perkawinan"),
        "pekerjaan": get_field("pekerjaan"),
        "kewarganegaraan": get_field("kewarganegaraan"),
        "kabupaten": get_field("kabupaten"),
        "provinsi": get_field("provinsi"),
        "kelurahan_desa": get_field("kelurahan_desa"),
        "kecamatan": get_field("kecamatan"),
        "raw": doc.text
    }
    return parsed


