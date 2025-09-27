# BISACare-AI

BISACare-AI adalah platform asisten AI untuk membantu proses administrasi dan klaim asuransi kesehatan di Indonesia. Sistem ini mengintegrasikan berbagai fitur berbasis AI seperti ekstraksi data dokumen, rekomendasi rumah sakit & asuransi, chatbot, transkripsi suara, dan pembuatan surat banding klaim.

## Fitur Utama

- **Scan Dokumen Otomatis**
  - Scan KTP, polis asuransi, diagnosis dokter, dan invoice rumah sakit menggunakan Google Document AI.
  - Ekstraksi otomatis data penting dari gambar dokumen.

- **Rekomendasi Rumah Sakit & Asuransi**
  - Rekomendasi rumah sakit dan produk asuransi berbasis semantic search (SBERT + FAISS).
  - Data rumah sakit dan asuransi sudah diproses dan diindeks untuk pencarian cepat.

- **BISAbot (Chatbot Asuransi)**
  - Chatbot berbasis Gemini Pro dengan Retrieval-Augmented Generation (RAG) dari dokumen asuransi.
  - Menjawab pertanyaan seputar asuransi dengan referensi dokumen asli.

- **AI Insurance Guide**
  - Panduan langkah demi langkah pembuatan asuransi dan penjelasan produk.

- **AI Coverage Display**
  - Analisis coverage/tanggungan asuransi berdasarkan diagnosis, polis, dan invoice.

- **AI Claim Denial Rewriter**
  - Membantu memahami alasan penolakan klaim dan memberi saran langkah banding.

- **Surat Aju Banding Otomatis**
  - Generate surat banding PDF otomatis berdasarkan data klaim dan alasan penolakan.

- **Transkripsi Suara**
  - Transkripsi otomatis file audio (m4a/mp3/wav) menggunakan model Whisper yang sudah di-finetune Bahasa Indonesia.

## Deployment AI
https://fastapi-ai-service-1081333106174.asia-southeast2.run.app/

## Dokumentasi API
https://fastapi-ai-service-1081333106174.asia-southeast2.run.app/docs#/


