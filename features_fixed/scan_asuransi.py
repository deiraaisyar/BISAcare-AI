import base64
import os
from google.cloud import documentai_v1 as documentai

GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "credential.json")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_APPLICATION_CREDENTIALS

PROJECT_ID = "1081333106174"
LOCATION = "us"
PROCESSOR_ID = "171bdbf2140012bd"

def scan_asuransi_pipeline(image_base64: str) -> dict:
    """Pipeline OCR Asuransi menggunakan Google Document AI"""
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

    def get_field(field_name):
        for entity in doc.entities:
            if field_name.lower() in entity.type_.lower():
                return entity.mention_text
        return None

    parsed = {
        "coverage": get_field("coverage"),
        "coverage_area": get_field("coverage_area"),
        "jenis_asuransi": get_field("jenis_asuransi"),
        "nama": get_field("nama"),
        "nama_asuransi": get_field("nama_asuransi"),
        "nomor_kartu": get_field("nomor_kartu"),
        "nomor_polis": get_field("nomor_polis"),
        "raw": doc.text
    }
    return parsed