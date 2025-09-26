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

def fallback_parse_raw_diagnosis(raw_text, field_name):
    if field_name == "diagnosa":
        match = re.search(
            r"terdiagnosa\s*([^.]+?)(?:terapi|obat|dengan|dan telah|$)",
            raw_text,
            re.IGNORECASE
        )
        if not match:
            match = re.search(
                r"diagnosa\s*([^.]+?)(?:terapi|obat|dengan|dan telah|$)",
                raw_text,
                re.IGNORECASE
            )
        if match:
            diagnosa_text = match.group(1)
            diagnosa_list = re.split(r",|dan", diagnosa_text)
            return [d.strip() for d in diagnosa_list if d.strip()]
        return []
    patterns = {
        "jenis_kelamin": r"Jenis Kelamin\s*[:\-]?\s*([^\n]+)",
        "nama_dokter": r"Nama\s*[:\-]?\s*(dr\.?\s*[^\n]+)",
        "nama_pasien": r"Nama\s*[:\-]?\s*([^\n]+)\nNIK",
        "nik": r"NIK\s*[:\-]?\s*([^\n]+)",
    }
    pattern = patterns.get(field_name)
    if pattern:
        match = re.search(pattern, raw_text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None

def scan_diagnosis_pipeline(image_base64: str) -> dict:
    client = get_docai_client()
    image_bytes = base64.b64decode(image_base64)

    name = f"projects/{PROJECT_ID}/locations/{LOCATION}/processors/{PROCESSOR_ID}"

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
        # Ambil entity dari Document AI
        entities = [entity.mention_text for entity in doc.entities if field_name.lower() in entity.type_.lower()]
        if entities:
            return entities
        # Fallback regex untuk diagnosa
        if field_name == "diagnosa":
            return fallback_parse_raw_diagnosis(doc.text, "diagnosa")
        return []

    def get_field(field_name):
        for entity in doc.entities:
            if field_name.lower() in entity.type_.lower():
                return entity.mention_text
        return fallback_parse_raw_diagnosis(doc.text, field_name)

    diagnosa_list = get_fields("diagnosa")
    penyakit_val = get_field("penyakit")
    if not penyakit_val and diagnosa_list:
        penyakit_val = diagnosa_list[0]  # ambil diagnosa pertama

    parsed = {
        "diagnosa": diagnosa_list,
        "jenis_kelamin": get_field("jenis_kelamin"),
        "nama_dokter": get_field("nama_dokter"),
        "nama_pasien": get_field("nama_pasien"),
        "nik": get_field("nik"),
        "penyakit": penyakit_val,
        "raw": doc.text
    }
    return parsed
