import base64
import os, json
import re
from google.cloud import documentai_v1 as documentai
from google.oauth2 import service_account

PROJECT_ID = "1081333106174"
LOCATION = "us"
PROCESSOR_ID = "d788d904b365af4"

# --- client setup ---
def get_docai_client():
    credentials = None
    gac = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

    if gac and gac.startswith("{"):
        creds_info = json.loads(gac)
        credentials = service_account.Credentials.from_service_account_info(creds_info)
    elif gac:
        credentials = service_account.Credentials.from_service_account_file(gac)
    else:
        credentials = service_account.Credentials.from_service_account_file("credential.json")

    return documentai.DocumentProcessorServiceClient(credentials=credentials)

# --- helper parse items ---
def parse_invoice_items(raw_text: str):
    """
    Cari daftar item & harga di dalam raw text invoice.
    """
    items = []
    prices = []

    lines = raw_text.splitlines()
    for i, line in enumerate(lines):
        line_clean = line.strip()

        # Item: uppercase atau ada nama obat/lab/visit
        if re.match(r"^[A-Z][A-Za-z0-9\s\-\.,]+$", line_clean) and not re.match(r"^\d+$", line_clean):
            # kalau baris berikutnya angka besar, berarti ini item
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip().replace('.', '').replace(',', '')
                if re.match(r"^\d{3,}$", next_line):
                    items.append(line_clean)

        # Harga: angka besar ribuan
        if re.match(r"^\d{1,3}(\.\d{3})+(,\d+)?$", line_clean) or re.match(r"^\d{4,}$", line_clean.replace('.', '').replace(',', '')):
            prices.append(line_clean)

    return items, prices

# --- fallback regex untuk field lain ---
def fallback_parse_raw_invoicers(raw_text, field_name):
    patterns = {
        "nama": r"Name\s*[:\-]?\s*([^\n]+)",
        "nama_rumah_sakit": r"(Siloam Hospitals [^\n]+|Hospitals\s+[^\n]+)",
        "nomor_invoice": r"Invoive No\s*[:\-]?\s*([^\n]+)|Invoice No\s*[:\-]?\s*([^\n]+)",
        "tanggal": r"Tanggal\s*[:\-]?\s*([^\n]+)|Invoice Date\s*[:\-]?\s*([^\n]+)",
        "total": r"TOTAL\s*[:\-]?\s*([^\n]+)|SUB TOTAL\s*[:\-]?\s*([^\n]+)",
    }
    pattern = patterns.get(field_name)
    if pattern:
        match = re.search(pattern, raw_text, re.IGNORECASE)
        if match:
            for group in match.groups():
                if group:
                    return group.strip()
    return None

# --- main pipeline ---
def scan_invoicers_pipeline(image_base64: str) -> dict:
    """Pipeline OCR Invoice Rumah Sakit menggunakan Google Document AI"""
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

    # Ambil dengan entity
    def get_fields(field_name):
        return [entity.mention_text for entity in doc.entities if field_name.lower() in entity.type_.lower()]

    def get_field(field_name):
        for entity in doc.entities:
            if field_name.lower() in entity.type_.lower():
                return entity.mention_text
        return fallback_parse_raw_invoicers(doc.text, field_name)

    # --- parse item list ---
    items, prices = parse_invoice_items(doc.text)

    parsed = {
        "items": get_fields("items") or items,
        "items_price": get_fields("items_price") or prices,
        "nama": get_field("nama"),
        "nama_rumah_sakit": get_field("nama_rumah_sakit"),
        "nomor_invoice": get_field("nomor_invoice"),
        "tanggal": get_field("tanggal"),
        "total": get_field("total"),
        "raw": doc.text
    }
    return parsed
