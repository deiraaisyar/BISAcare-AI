# BISACare-AI

## Cara Install & Menjalankan Project

### 1. Clone Repository
```bash
git clone https://github.com/username/NLXOTI-AI.git
cd NLXOTI-AI
```

### 2. Buat dan Aktifkan Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Jalankan Backend FastAPI
```bash
uvicorn main:app --reload
```
Akses API di [http://localhost:8000/docs](http://localhost:8000/docs)

---

## API Endpoints

### 1. `/bisabot` (POST)
Chat dengan BISAbot (chatbot asuransi, RAG + Gemini).

**Input:**  
```json
{ "question": "Tulis pertanyaan Anda di sini" }
```
**Output:**  
```json
{ "answer": "Jawaban dari BISAbot" }
```

---

### 2. `/bisabot/history` (GET)
Ambil riwayat chat BISAbot.

**Output:**  
```json
{ "history": [ ... ] }
```

---

### 3. `/bisabot/history` (DELETE)
Hapus seluruh riwayat chat BISAbot.

**Output:**  
```json
{ "message": "Chat history cleared successfully" }
```

---

### 4. `/surat_aju_banding` (POST)
Generate surat aju banding asuransi (PDF).

**Input:**  
```json
{
  "nama": "...",
  "nomor_polis": "...",
  "alamat": "...",
  "nomor_hp": "...",
  "tanggal_pengajuan": "...",
  "nomor_klaim": "...",
  "perihal_klaim": "...",
  "alasan_penolakan": "...",
  "alasan_banding": "...",
  "nama_asuransi": "..."
}
```
**Output:**  
```json
{
  "message": "Surat aju banding berhasil dibuat",
  "filename": "...pdf",
  "download_url": "/download/..."
}
```

---

### 5. `/rekomendasi_rumah_sakit` (POST)
Rekomendasi rumah sakit berdasarkan data user.

**Input:**  
```json
{
  "nama": "...",
  "kelurahan_desa": "...",
  "kecamatan": "...",
  "jenis_layanan": "...",
  "keluhan": "...",
  "nama_asuransi": "...",
  "nama_provinsi": "...",
  "nama_daerah": "...",
  "top_n": 5
}
```
**Output:**  
```json
{ "results": [ ... ] }
```

---

### 6. `/rekomendasi_asuransi` (POST)
Rekomendasi produk asuransi.

**Input:**  
```json
{ "query": "...", "top_n": 5 }
```
**Output:**  
```json
{ "results": [ ... ] }
```

---

### 7. `/download/{filename}` (GET)
Download file PDF hasil generate surat aju banding.

---

### 8. `/isi_data` (POST)
Upload foto KTP & Polis, plus data form lain.

**Input:**  
- Form-data:
  - `foto_ktp` (file)
  - `foto_polis` (file)
  - `nomor_polis` (string)
  - `jenis_layanan` (string)
  - `nomor_hp` (string)
  - `input_keluhan` (string)

**Output:**  
```json
{
  "ktp": { ... },
  "polis": { ... },
  "raw_text": "...",
  "nomor_polis": "...",
  "layanan": "...",
  "nomor_hp": "...",
  "keluhan": "..."
}
```

---

### 9. `/bantu_proses_ai` (POST)
Cek data hasil isi_data dan memberi saran AI.

**Input:**  
```json
{ ...isi_data }
```
**Output:**  
```json
{
  "status": "cek_data",
  "saran": [ ... ],
  "data_isi": { ... },
  "chat_history": [ ... ]
}
```

---

### 10. `/slip_rumah_sakit` (POST)
Upload slip rumah sakit, ekstrak data penting dengan AI.

**Input:**  
- Form-data: `foto_slip` (file)

**Output:**  
```json
{
  "slip_id": "...",
  "filename": "...",
  "parsed": { ... }
}
```

---

### 11. `/slip_rumah_sakit/{slip_id}` (GET)
Ambil hasil ekstraksi slip rumah sakit berdasarkan slip_id.

**Output:**  
```json
{
  "filename": "...",
  "raw_text": "...",
  "parsed": { ... }
}
```

---

### 12. `/keluhanmu_bisa_diklaim` (POST)
Analisis keluhan kesehatan, input bisa teks atau audio.

**Input:**  
- Form-data:
  - `keluhan_text` (string, optional)
  - `metode_input` (string, default: "text" atau "voice")
  - `audio_file` (file, optional)

**Output:**  
```json
{
  "status": "success",
  "keluhan_id": "...",
  "keluhan_input": "...",
  "metode_input": "...",
  "analisis": {
    "persentase_kemungkinan_klaim": "...",
    "kemungkinan_diagnosis": [...],
    "rekomendasi_tindakan": [...],
    "tingkat_urgensi": "...",
    "dokumen_pendukung_diperlukan": [...]
  },
  "disclaimer": "..."
}
```

---

### 13. `/keluhanmu_bisa_diklaim/{keluhan_id}` (GET)
Ambil data keluhan berdasarkan ID.

**Output:**  
```json
{
  "keluhan_input": "...",
  "metode_input": "...",
  "analisis": { ... },
  "disclaimer": "..."
}
```

---

### 14. `/hasil_diagnosis_dokter` (POST)
Terima hasil diagnosis dokter melalui foto, text, atau audio.

**Input:**  
- Form-data:
  - `foto_diagnosis` (file, optional)
  - `diagnosis_text` (string, optional)
  - `diagnosis_audio` (file, optional)

**Output:**  
```json
{
  "status": "success",
  "hasil_diagnosis": {
    "foto_diagnosis": "...hasil OCR...",
    "diagnosis_text": "...text manual...",
    "diagnosis_audio": "...hasil transkripsi audio..."
  }
}
```

---

### 15. `/tanggungan_ai` (POST)
Analisis tanggungan asuransi berdasarkan data isi_data dan hasil diagnosis dokter (menggunakan Gemini/AI).

**Input:**  
```json
{
  "isi_data": { ... },
  "hasil_diagnosis": { ... }
}
```
**Output:**  
```json
{
  "status": "success",
  "analisis_tanggungan": "...",
  "isi_data": { ... },
  "hasil_diagnosis": { ... }
}
```

---

### 16. `/scan_data_slip` (POST)
Upload foto slip rumah sakit (gambar) dan/atau audio slip (suara), baca teks slip dengan OCR dan transkripsi suara.

**Input:**  
- Form-data:
  - `foto_slip` (file, optional)
  - `audio_slip` (file, optional)

**Output:**  
```json
{
  "slip_text": "...",           // hasil OCR gambar slip
  "slip_audio_text": "..."      // hasil transkripsi audio slip
}
```

---

### 17. `/` (GET)
Root endpoint, menampilkan deskripsi singkat API dan daftar fitur.

---

## Catatan Setup

- Letakkan file PDF asuransi untuk RAG di folder: `./rag/documents/`
- Format audio yang didukung: mp3, wav, m4a, flac, ogg, webm, mp4
- Untuk OCR, pastikan Tesseract sudah terinstall di