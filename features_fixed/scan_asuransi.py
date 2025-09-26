import base64
import os
import json
from google.cloud import documentai_v1 as documentai
from google.oauth2 import service_account

PROJECT_ID = "1081333106174"
LOCATION = "us"
PROCESSOR_ID = "171bdbf2140012bd"

def get_docai_client():
    gac = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if gac and gac.startswith("{"):
        creds_info = json.loads(gac)
        credentials = service_account.Credentials.from_service_account_info(creds_info)
    elif gac:
        credentials = service_account.Credentials.from_service_account_file(gac)
    else:
        credentials = service_account.Credentials.from_service_account_file("credential.json")
    return documentai.DocumentProcessorServiceClient(credentials=credentials)

def scan_asuransi_pipeline(image_base64: str) -> dict:
    """Pipeline OCR Asuransi menggunakan Google Document AI"""
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