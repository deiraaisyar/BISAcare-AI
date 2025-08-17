# BISACare-AI

## API Endpoints

### 1. `/bisabot` (POST)
**Deskripsi:**  
Chat dengan BISAbot (chatbot asuransi) yang sudah terintegrasi dengan RAG.

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

### 4. `/surat_aju_banding` (POST)
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

### 5. `/rekomendasi_rumah_sakit` (POST)
**Deskripsi:**  
Rekomendasi rumah sakit berdasarkan data user dan asuransi.

**Input:**  
```json
{
  "nama": "Nama",
  "kelurahan_desa": "Kelurahan/Desa",
  "kecamatan": "Kecamatan",
  "jenis_layanan": "Rawat Inap/Rawat Jalan",
  "keluhan": "Keluhan kesehatan",
  "nama_asuransi": "Nama Asuransi",
  "nama_provinsi": "Nama Provinsi",
  "nama_daerah": "Nama Daerah",
  "top_n": 5
}
```
**Output:**  
```json
{
  "results": [ ... ]
}
```

---

### 6. `/rekomendasi_asuransi` (POST)
**Deskripsi:**  
Rekomendasi produk asuransi berdasarkan query user.

**Input:**  
```json
{
  "query": "Saya ingin asuransi kesehatan keluarga",
  "top_n": 5
}
```
**Output:**  
```json
{
  "results": [ ... ]
}
```

---

### 7. `/download/{filename}` (GET)
**Deskripsi:**  
Download file PDF hasil generate surat aju banding.

**Input:**  
- Path parameter: `filename` (nama file PDF)

**Output:**  
- File PDF sebagai response.

---

### 8. `/isi_data` (POST)
**Deskripsi:**  
Upload foto KTP & Polis, plus data form lain.  
Output: hasil OCR & parsing + data form.

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
  "jenis_layanan": "...",
  "nomor_hp": "...",
  "keluhan": "..."
}
```

---

### 9. `/bantu_proses_ai` (POST)
**Deskripsi:**  
Cek data hasil isi_data dan memberi saran AI untuk langkah selanjutnya.

**Input:**  
```json
{
  "ktp": { ... },
  "polis": { ... },
  "nomor_polis": "...",
  "jenis_layanan": "...",
  "nomor_hp": "...",
  "keluhan": "..."
}
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
**Deskripsi:**  
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
**Deskripsi:**  
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
**Deskripsi:**  
Analisis keluhan kesehatan, input bisa teks atau audio/video.

**Input:**  
- Form-data:
  - `keluhan_text` (string, optional)
  - `metode_input` (string, default: "text")
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
**Deskripsi:**  
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

### 14. `/` (GET)
**Deskripsi:**  
Root endpoint, menampilkan deskripsi singkat API dan daftar fitur.

**Output:**  
```json
{
  "message": "Welcome to BISAcare - AI-Powered Insurance Assistant",
  "features": { ... },
  "endpoints": { ... },
  "setup": { ... }
}
```

---

## Catatan Setup

- Letakkan file PDF asuransi untuk RAG di folder: `./rag/documents/`
- Format audio yang didukung: mp3, wav, m4a, flac, ogg, webm, mp4

---