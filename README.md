# BISACare-AI

BISACare-AI is an AI assistant platform designed to simplify health insurance administration and claims processing in Indonesia. The project integrates document extraction, recommendation systems, a chatbot with retrieval-augmented generation (RAG), speech-to-text, and automatic claim appeal (surat aju banding) generation.

## Features

- **Automated Document Scanning & Extraction** — KTP, insurance policies, medical diagnoses, and hospital invoices via Google Document AI pipelines.
- **Hospital & Insurance Recommendation** — Semantic search using Sentence-BERT embeddings and FAISS indexes for fast, relevant recommendations.
- **BISAbot (Chatbot)** — Conversational assistant with RAG over indexed insurance documents.
- **AI Insurance Guide** — Step-by-step guidance on insurance procedures and product explanations.
- **Coverage Analysis** — Automated coverage/benefit analysis from diagnosis, policy and invoice data.
- **Claim Denial Assistant** — Explain claim denials and suggest appeal strategies.
- **Appeal Letter Generator** — Generate PDF appeal letters (`surat_aju_banding`) programmatically.
- **Voice Transcription** — Transcribe audio (m4a/mp3/wav) using a Whisper-based model.

## Repository structure (important files)

Key files and folders:

- `main_fixed.py` — FastAPI application exposing endpoints for all features.
- `features_fixed/` — Feature implementations: `bisabot.py`, `scan_ktp.py`, `scan_asuransi.py`, `scan_diagnosis.py`, `scan_invoicers.py`, `voice_transcribe.py`, `hospital_recommender.py`, `insurance_recommender.py`, `generate_ajubanding.py`, etc.
- `daftar_rumah_sakit/` and `daftar_asuransi/` — Data processing, preprocessing scripts and preprocessed data (CSV/JSON) used for recommendations.
- `rag/` — RAG loader and retriever utilities.
- `schema_fixed/schemas.py` — Pydantic request/response models used by the API.
- `utils/gcs_utils.py` — Google Cloud Storage helper (upload logic).
- `requirements.txt` — Python dependencies.
- `Dockerfile` — containerization instructions.
- `test_uploadfile/test_code/` — example scripts that POST test files to deployed endpoints.

## Quickstart — Local development

1. Create and activate a Python virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set environment variables (best via a `.env` file):

```
GOOGLE_APPLICATION_CREDENTIALS=./credential.json
HF_TOKEN=<your_huggingface_token>
# other keys as needed
```

4. Run the API:

```bash
uvicorn main_fixed:app --reload --host 0.0.0.0 --port 8000
```

The API docs will be available at `http://localhost:8000/docs`.

## Docker

Build and run with Docker (image uses Python 3.11-slim):

```bash
docker build -t bisacare-ai .
docker run -p 8080:8080 --env-file .env bisacare-ai
```

## Key Endpoints

- `POST /surat-aju-banding` — Generate an appeal letter PDF. Returns a `download_url` (public GCS URL) after the file is uploaded.
- `POST /scan-ktp` — Upload an image and extract KTP fields.
- `POST /scan-asuransi` — Upload an insurance policy image and extract data.
- `POST /scan-diagnosis` — Upload a diagnosis image and extract relevant fields.
- `POST /scan-invoicers` — Upload an invoice image and extract invoice items.
- `POST /transcribe` — Upload audio to transcribe.
- `POST /rekomendasi-rumah-sakit` — Request hospital recommendations.
- `POST /rekomendasi-asuransi` — Request insurance product recommendations.

See `schema_fixed/schemas.py` for full request/response shapes used by endpoints.

## GCS (Google Cloud Storage) integration

- The project uploads generated PDFs (e.g. `surat_aju_banding_...pdf`) to a GCS bucket and returns a public URL to the frontend.
- Credentials: provide a service account JSON file and point `GOOGLE_APPLICATION_CREDENTIALS` to it (or set the content in the env var if you prefer). The code in `main_fixed.py` and `features_fixed/*` uses the same credential-loading approach as Document AI utilities.
- Bucket settings: If you enable Uniform bucket-level access (UBLA) on the bucket, do NOT call `blob.make_public()` in code. Instead, make the bucket objects readable by granting the `Storage Object Viewer` role to `allUsers` in the bucket's Permissions tab. Example returned URL format:

```
https://storage.googleapis.com/<bucket-name>/<path-to-object>
```

### Example: appeal generation flow

1. Frontend POSTs JSON to `/surat-aju-banding` with required fields (name, policy number, claim details, reasons, etc.).
2. Backend generates PDF locally, uploads the file to GCS using `utils/gcs_utils.py` or `upload_to_gcs(...)`, and returns JSON:

```json
{
  "message": "Surat aju banding berhasil dibuat",
  "filename": "surat_aju_banding_John_Doe_20251122010101.pdf",
  "download_url": "https://storage.googleapis.com/aju-banding-pdf-result/aju-banding/surat_aju_banding_John_Doe_20251122010101.pdf"
}
```

3. Frontend can open or download the returned `download_url` directly (no backend GET required), e.g.: `window.open(download_url, '_blank')`.

## Notes and troubleshooting

- If you see errors when calling `blob.make_public()` like `Cannot get legacy ACL for an object when uniform bucket-level access is enabled`, remove `blob.make_public()` and instead set the bucket-level permissions as described above.
- Verify `credential.json` or `GOOGLE_APPLICATION_CREDENTIALS` is present and the service account has `Storage Object Creator` (and optionally `Storage Object Viewer`) on the target bucket.
- For Document AI features, ensure the service account has access to the Document AI processor and the correct `PROJECT_ID`/`PROCESSOR_ID` are set in `features_fixed/*` files.

## Tests

- Basic test scripts are provided in `test_uploadfile/test_code/` (examples that POST files to the live API). You can run them as Python scripts after starting the server locally or by pointing them at a deployed service URL.

## Contributing

- Open an issue for bugs or feature requests.
- Prefer small pull requests with focused changes.

## Deployment AI
https://fastapi-ai-service-1081333106174.asia-southeast2.run.app/


