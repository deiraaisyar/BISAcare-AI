import base64
import os
from google.cloud import documentai_v1 as documentai

GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "credential.json")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_APPLICATION_CREDENTIALS

PROJECT_ID = "1081333106174"
LOCATION = "us"
PROCESSOR_ID = "7e017aeeb30ab9b7"

def scan_diagnosis_pipeline(image_base64: str) -> dict:
    """Pipeline OCR Diagnosis menggunakan Google Document AI"""
    client = documentai.DocumentProcessorServiceClient()
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

    def get_fields(field_name):
        # Ambil semua entity dengan nama field_name (untuk list)
        return [entity.mention_text for entity in doc.entities if field_name.lower() in entity.type_.lower()]

    def get_field(field_name):
        for entity in doc.entities:
            if field_name.lower() in entity.type_.lower():
                return entity.mention_text
        return None

    parsed = {
        "diagnosa": get_fields("diagnosa"),
        "jenis_kelamin": get_field("jenis_kelamin"),
        "nama_dokter": get_field("nama_dokter"),
        "nama_pasien": get_field("nama_pasien"),
        "nik": get_field("nik"),
        "penyakit": get_field("penyakit"),
        "raw": doc.text
    }
    return parsed