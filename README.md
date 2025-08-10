# BISACare-AI

## API Endpoints

### 1. `/bisabot` (POST)
**Deskripsi:**  
Chat dengan BISAbot (chatbot asuransi) yang sudah terintegrasi dengan RAG (Retrieval Augmented Generation).

**Input:**  
```json
{
  "question": "Tulis pertanyaan Anda di sini"
}
```

**Output:**  
```json
{
  "answer": "Jawaban dari BISAbot"
}
```

---

### 2. `/bisabot/history` (GET)
**Deskripsi:**  
Mengambil riwayat chat BISAbot.

**Output:**  
```json
{
  "history": [
    {"role": "user", "content": "..."},
    {"role": "bot", "content": "..."}
  ]
}
```

---

### 3. `/bisabot/history` (DELETE)
**Deskripsi:**  
Menghapus seluruh riwayat chat BISAbot.

**Output:**  
```json
{
  "message": "Chat history cleared successfully"
}
```

---

### 4. `/bisabot/status` (GET)
**Deskripsi:**  
Cek status BISAbot dan status RAG (apakah index sudah siap).

**Output:**  
```json
{
  "bisabot": "online",
  "rag": "ready" // atau status lain
}
```

---

### 5. `/surat_aju_banding` (POST)
**Deskripsi:**  
Generate surat aju banding asuransi dalam format PDF.

**Input:**  
```json
{
  "nama": "Nama Lengkap",
  "no_polis": "Nomor Polis",
  "alamat": "Alamat Lengkap",
  "no_telepon": "Nomor Telepon",
  "tanggal_pengajuan": "YYYY-MM-DD",
  "nomor_klaim": "Nomor Klaim",
  "perihal_klaim": "Perihal Klaim",
  "alasan_penolakan": "Alasan Penolakan Klaim",
  "alasan_banding": "Alasan Banding",
  "nama_perusahaan_asuransi": "Nama Perusahaan Asuransi"
}
```

**Output:**  
```json
{
  "message": "Surat aju banding berhasil dibuat",
  "filename": "surat_aju_banding_xxxxxxxx.pdf",
  "download_url": "/download/surat_aju_banding_xxxxxxxx.pdf"
}
```

---

### 6. `/keluhanmu_bisa_diklaim` (POST)
**Deskripsi:**  
Analisis keluhan kesehatan berbasis teks untuk estimasi kemungkinan klaim asuransi.

**Input:**  
```json
{
  "keluhan_text": "Keluhan kesehatan Anda",
  "metode_input": "text"
}
```

**Output:**  
```json
{
  "status": "success",
  "keluhan_input": "...",
  "metode_input": "text",
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

### 7. `/keluhanmu_bisa_diklaim/voice` (POST)
**Deskripsi:**  
Analisis keluhan kesehatan berbasis audio (voice note atau video).

**Input:**  
- Form-data dengan field: `audio_file` (file audio/video, format: mp3, wav, m4a, flac, ogg, webm, mp4)

**Output:**  
```json
{
  "status": "success",
  "keluhan_input": "...",
  "metode_input": "voice",
  "original_filename": "...",
  "file_type": "...",
  "transcribed_text": "...",
  "analisis": { ... },
  "disclaimer": "..."
}
```

---

### 8. `/download/{filename}` (GET)
**Deskripsi:**  
Download file PDF hasil generate surat aju banding.

**Input:**  
- Path parameter: `filename` (nama file PDF)

**Output:**  
- File PDF sebagai response.

---

### 9. `/` (GET)
**Deskripsi:**  
Root endpoint, menampilkan deskripsi singkat API dan daftar fitur.

---

## Catatan Setup

- Letakkan file PDF asuransi untuk RAG di folder: `./rag/documents/`
- Format audio yang didukung: mp3, wav, m4a, flac, ogg, webm, mp4

---