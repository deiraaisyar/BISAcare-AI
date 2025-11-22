# cURL examples for BISACare-AI endpoints

Replace `{{BASE_URL}}` with your API base (e.g. `http://localhost:8000` or your deployed URL).

1) Generate appeal PDF (`/surat-aju-banding`)

```bash
curl -X POST "{{BASE_URL}}/surat-aju-banding" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "John Doe",
    "policy_number": "POL123456",
    "claim_number": "CLM987654",
    "date": "2025-11-22",
    "issue_summary": "Claim rejected due to missing documentation",
    "appeal_reason": "I provide doctor notes and invoice attached",
    "additional_notes": "Please review the attached documents"
  }'
```

Response example (JSON):

```json
{
  "message": "Surat aju banding berhasil dibuat",
  "filename": "surat_aju_banding_John_Doe_20251122010101.pdf",
  "download_url": "https://storage.googleapis.com/your-bucket/aju-banding/..pdf"
}
```

2) Scan KTP (image upload to `/scan-ktp`)

```bash
curl -X POST "{{BASE_URL}}/scan-ktp" \
  -H "Accept: application/json" \
  -F "file=@/path/to/ktp.jpg;type=image/jpeg"
```

3) Transcribe audio (`/transcribe`)

```bash
curl -X POST "{{BASE_URL}}/transcribe" \
  -F "file=@/path/to/audio.wav;type=audio/wav" \
  -F "language=id"
```

4) Scan diagnosis (image upload)

```bash
curl -X POST "{{BASE_URL}}/scan-diagnosis" \
  -F "file=@/path/to/diagnosis.jpg;type=image/jpeg"
```

Notes:
- For multipart uploads include `-F` with the `file=@...` form field.
- If your API requires auth, add `-H "Authorization: Bearer $TOKEN"`.
