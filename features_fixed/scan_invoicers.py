import base64
import os
from google.cloud import documentai_v1 as documentai

GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "credential.json")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_APPLICATION_CREDENTIALS

PROJECT_ID = "1081333106174"
LOCATION = "us"
PROCESSOR_ID = "65f7ada432f057b6"

def scan_invoicers_pipeline(image_base64: str) -> dict:
    """Pipeline OCR Invoice Rumah Sakit menggunakan Google Document AI"""
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
        "items": get_fields("items"),
        "items_price": get_fields("items_price"),
        "nama": get_field("nama"),
        "nama_rumah_sakit": get_field("nama_rumah_sakit"),
        "nomor_invoice": get_field("nomor_invoice"),
        "tanggal": get_field("tanggal"),
        "total": get_field("total"),
        "raw": doc.text
    }
    return parsed