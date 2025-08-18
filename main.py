from fastapi import FastAPI, Request, HTTPException, File, UploadFile, Form, Body
from fastapi.responses import FileResponse
from pydantic import BaseModel
from features.bisabot.bisabot import ask_bisabot, get_chat_history, clear_chat_history
from features.surat_aju_banding.surat_aju_banding import buat_surat_aju_banding_pdf
from features.keluhanmu_bisa_diklaim.keluhanmu_bisa_diklaim import analyze_health_complaint, analyze_health_complaint_from_audio
from features.hospital_recommender.hospital_recommender import recommend_hospitals
from features.data_asuransi_ai.scan_data import extract_text, parse_with_ai
from features.bantu_proses_ai.bantu_proses_ai import cek_data_isi_data
from features.slip_rumah_sakit.slip_rumah_sakit import extract_text, parse_slip_with_ai
from features.insurance_recommender.insurance_recommender import load_asuransi_data, recommend_asuransi
from features.hasil_diagnosis_dokter.hasil_diagnosis_dokter import process_diagnosis
from features.tanggungan_ai.tanggungan_ai import analisis_tanggungan_ai
from daftar_rumah_sakit.data_processing import load_faiss_index, load_json, build_model
import os
import uuid
import tempfile
import shutil
from typing import Optional
import logging
from features.data_asuransi_ai_slip.data_asuransi_ai_slip import extract_slip_text
import whisper

app = FastAPI(title="BISAcare - AI-Powered Insurance Assistant")
logger = logging.getLogger("uvicorn.error")

class Query(BaseModel):
    question: str

class SuratAjuBandingRequest(BaseModel):
    nama: str
    nomor_polis: str           # Ubah dari no_polis
    alamat: str
    nomor_hp: str              # Ubah dari no_telepon
    tanggal_pengajuan: str
    nomor_klaim: str
    perihal_klaim: str
    alasan_penolakan: str
    alasan_banding: str
    nama_asuransi: str         # Ubah dari nama_perusahaan_asuransi

class KeluhanRequest(BaseModel):
    keluhan_text: str
    metode_input: str = "text"

class HospitalRecommendRequest(BaseModel):
    nama: str
    kelurahan_desa: str
    kecamatan: str
    jenis_layanan: str
    keluhan: str
    nama_asuransi: str
    nama_provinsi: str
    nama_daerah: str
    top_n: int = 5

class InsuranceRecommendRequest(BaseModel):
    query: str
    top_n: int = 5

@app.post("/bisabot") #OK
async def chat(query: Query):
    """Chat dengan BISAbot yang sudah terintegrasi dengan RAG"""
    response = ask_bisabot(query.question)
    return {"answer": response}

@app.get("/bisabot/history") #OK
async def get_history():
    """Get chat history"""
    history = get_chat_history()
    return {"history": history}

@app.delete("/bisabot/history") #OK
async def clear_history():
    """Clear chat history"""
    clear_chat_history()
    return {"message": "Chat history cleared successfully"}

@app.post("/surat_aju_banding") #OK
async def buat_surat_banding(request: SuratAjuBandingRequest):
    try:
        unique_id = str(uuid.uuid4())[:8]
        nama_file = f"surat_aju_banding_{unique_id}.pdf"
        
        buat_surat_aju_banding_pdf(
            nama=request.nama,
            no_polis=request.nomor_polis,              
            alamat=request.alamat,
            no_telepon=request.nomor_hp,              
            tanggal_pengajuan=request.tanggal_pengajuan,
            nomor_klaim=request.nomor_klaim,
            perihal_klaim=request.perihal_klaim,
            alasan_penolakan=request.alasan_penolakan,
            alasan_banding=request.alasan_banding,
            nama_perusahaan_asuransi=request.nama_asuransi, 
            nama_file_output=nama_file
        )
        
        if not os.path.exists(nama_file):
            raise HTTPException(status_code=500, detail="Gagal membuat file PDF")
        
        return {
            "message": "Surat aju banding berhasil dibuat",
            "filename": nama_file,
            "download_url": f"/download/{nama_file}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

DATA_PATH = "daftar_rumah_sakit/preprocessed/daftar_rumah_sakit_all.json"
INDEX_PATH = "daftar_rumah_sakit/app/embeddings/hospital_st.index"
MODEL_PATH = "daftar_rumah_sakit/app/models/st_model"

hospital_data = load_json(DATA_PATH)
hospital_index = load_faiss_index(INDEX_PATH)
hospital_model = build_model(MODEL_PATH)

@app.post("/rekomendasi_rumah_sakit") #OK
async def rekomendasi_rumah_sakit(request: HospitalRecommendRequest):
    try:
        results = recommend_hospitals(
            data=hospital_data,
            index=hospital_index,
            model=hospital_model,
            nama=request.nama,
            kelurahan_desa=request.kelurahan_desa,
            kecamatan=request.kecamatan,
            jenis_layanan=request.jenis_layanan,
            keluhan=request.keluhan,
            nama_asuransi=request.nama_asuransi,
            nama_provinsi=request.nama_provinsi,
            nama_daerah=request.nama_daerah,
            top_n=request.top_n
        )
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

ASURANSI_DATA_PATH = "daftar_asuransi/preprocessed/daftar_asuransi_all.json"
ASURANSI_INDEX_PATH = "daftar_asuransi/app/embeddings/asuransi_st.index"
ASURANSI_MODEL_PATH = "daftar_asuransi/app/models/st_model"

asuransi_data = load_json(ASURANSI_DATA_PATH)
asuransi_index = load_faiss_index(ASURANSI_INDEX_PATH)
asuransi_model = build_model(ASURANSI_MODEL_PATH)

@app.post("/rekomendasi_asuransi") #OK
async def rekomendasi_asuransi(request: InsuranceRecommendRequest):
    try:
        results = recommend_asuransi(
            query=request.query,
            data=asuransi_data,
            index=asuransi_index,
            model=asuransi_model,
            top_n=request.top_n
        )
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# Download surat aju banding
@app.get("/download/{filename}") #OK
async def download_file(filename: str):
    file_path = f"./{filename}"
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File tidak ditemukan")
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type='application/pdf'
    )

@app.post("/isi_data")
async def isi_data(
    foto_ktp: UploadFile = File(...),
    foto_polis: UploadFile = File(...),
    nomor_polis: str = Form(...),
    jenis_layanan: str = Form(...), 
    nomor_hp: str = Form(...),
    input_keluhan: str = Form(...)
):
    """
    Upload foto KTP & Polis, plus data form lain.
    Output: hasil OCR & parsing + data form.
    """
    ktp_bytes = await foto_ktp.read()
    ktp_raw_text = extract_text(ktp_bytes)
    ktp_parsed = parse_with_ai(ktp_raw_text)

    polis_bytes = await foto_polis.read()
    polis_raw_text = extract_text(polis_bytes)
    polis_parsed = parse_with_ai(polis_raw_text)

    if isinstance(polis_parsed, dict) and "jenis_layanan" in polis_parsed:
        polis_parsed.pop("jenis_layanan")

    result = {
        "ktp": ktp_parsed.get("ktp") if hasattr(ktp_parsed, "get") else ktp_parsed,
        "polis": polis_parsed.get("polis") if hasattr(polis_parsed, "get") else polis_parsed,
        "raw_text": f"{ktp_raw_text}\n{polis_raw_text}",
        "nomor_polis": nomor_polis,
        "layanan": jenis_layanan,
        "nomor_hp": nomor_hp,
        "keluhan": input_keluhan
    }
    return result

@app.post("/bantu_proses_ai") #OK
async def bantu_proses_ai_endpoint(
    data_isi: dict = Body(...)
):
    """
    Mengecek data hasil isi_data dan memberi saran AI untuk langkah selanjutnya.
    """
    result = cek_data_isi_data(data_isi)
    return result

slip_data_store = {}

@app.post("/slip_rumah_sakit")
async def upload_slip_rumah_sakit(
    foto_slip: UploadFile = File(...)
):
    """
    Upload foto slip rumah sakit, ekstrak data penting dengan AI.
    """
    image_bytes = await foto_slip.read()
    raw_text = extract_text(image_bytes)
    parsed = parse_slip_with_ai(raw_text)
    slip_id = str(uuid.uuid4())[:8]
    slip_data_store[slip_id] = {
        "filename": foto_slip.filename,
        "raw_text": raw_text,
        "parsed": parsed
    }
    return {
        "slip_id": slip_id,
        "filename": foto_slip.filename,
        "parsed": parsed
    }

@app.get("/slip_rumah_sakit/{slip_id}")
async def get_slip_rumah_sakit(slip_id: str):
    """
    Ambil hasil ekstraksi slip rumah sakit berdasarkan slip_id.
    """
    data = slip_data_store.get(slip_id)
    if not data:
        raise HTTPException(status_code=404, detail="Slip tidak ditemukan")
    return data

keluhan_data_store = {}

@app.post("/keluhanmu_bisa_diklaim") #OK
async def analisis_keluhan(
    keluhan_text: Optional[str] = Form(None),
    metode_input: Optional[str] = Form("text"),
    audio_file: Optional[UploadFile] = File(None)
):
    """
    Analisis keluhan kesehatan, input bisa teks atau audio/video.
    """
    try:
        if metode_input == "voice" and audio_file is not None:
            allowed_extensions = ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.webm', '.mp4']
            file_extension = os.path.splitext(audio_file.filename)[1].lower()
            if file_extension not in allowed_extensions:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Format file tidak didukung. Gunakan: {', '.join(allowed_extensions)}"
                )
            logger.info(f"Saving audio file: {audio_file.filename}")
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                shutil.copyfileobj(audio_file.file, temp_file)
                temp_file_path = temp_file.name
            logger.info(f"Temp file saved at: {temp_file_path}, exists: {os.path.exists(temp_file_path)}")
            try:
                result = analyze_health_complaint_from_audio(temp_file_path)
                keluhan_input = result.get("transcribed_text", "")
                metode = "voice" if file_extension != '.mp4' else "video"
            finally:
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
        elif keluhan_text is not None and keluhan_text.strip():
            result = analyze_health_complaint(keluhan_text)
            keluhan_input = keluhan_text
            metode = "text"
        else:
            raise HTTPException(status_code=400, detail="Keluhan tidak boleh kosong")
        
        keluhan_id = str(uuid.uuid4())[:8]
        keluhan_data_store[keluhan_id] = {
            "keluhan_input": keluhan_input,
            "metode_input": metode,
            "analisis": {
                "persentase_kemungkinan_klaim": result.get("persentase_klaim", "Tidak dapat ditentukan"),
                "kemungkinan_diagnosis": result.get("kemungkinan_diagnosis", []),
                "rekomendasi_tindakan": result.get("rekomendasi_tindakan", []),
                "tingkat_urgensi": result.get("tingkat_urgensi", "sedang"),
                "dokumen_pendukung_diperlukan": result.get("dokumen_pendukung", [])
            },
            "disclaimer": "Hasil analisis ini hanya sebagai referensi. Konsultasikan dengan dokter untuk diagnosis yang akurat."
        }
        return {
            "status": "success",
            "keluhan_id": keluhan_id,
            **keluhan_data_store[keluhan_id]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error menganalisis keluhan: {str(e)}")

@app.get("/keluhanmu_bisa_diklaim/{keluhan_id}") #OK
async def get_keluhanmu_bisa_diklaim(keluhan_id: str):
    data = keluhan_data_store.get(keluhan_id)
    if not data:
        raise HTTPException(status_code=404, detail="Data keluhan tidak ditemukan")
    return data

@app.post("/hasil_diagnosis_dokter")
async def hasil_diagnosis_dokter(
    foto_diagnosis: UploadFile = File(None),
    diagnosis_text: str = Form(None),
    diagnosis_audio: UploadFile = File(None)
):
    """
    Terima hasil diagnosis dokter melalui foto, text, atau voice over.
    """
    result = {}

    # 1. Jika ada foto diagnosis
    if foto_diagnosis is not None:
        image_bytes = await foto_diagnosis.read()
        # Proses OCR atau parsing gambar di sini
        result["foto_diagnosis"] = process_diagnosis(image_bytes=image_bytes)

    # 2. Jika ada text diagnosis
    if diagnosis_text is not None and diagnosis_text.strip():
        # Proses text diagnosis di sini
        result["diagnosis_text"] = process_diagnosis(text=diagnosis_text)

    # 3. Jika ada audio diagnosis
    if diagnosis_audio is not None:
        audio_path = f"/tmp/{diagnosis_audio.filename}"
        with open(audio_path, "wb") as f:
            f.write(await diagnosis_audio.read())
        # Proses transkripsi audio di sini
        result["diagnosis_audio"] = process_diagnosis(audio_path=audio_path)
        os.remove(audio_path)

    if not result:
        raise HTTPException(status_code=400, detail="Harus upload foto, isi text, atau voice diagnosis dokter.")

    return {
        "status": "success",
        "hasil_diagnosis": result
    }

@app.post("/tanggungan_ai")
async def tanggungan_ai_endpoint(
    isi_data: dict = Body(...),
    hasil_diagnosis: dict = Body(...)
):
    """
    Analisis tanggungan asuransi berdasarkan data isi_data dan hasil diagnosis dokter (menggunakan Gemini/AI).
    """
    result = analisis_tanggungan_ai(isi_data=isi_data, hasil_diagnosis=hasil_diagnosis)
    return result

@app.post("/scan_data_slip")
async def scan_data_slip(
    foto_slip: UploadFile = File(None),
    audio_slip: UploadFile = File(None)
):
    """
    Upload foto slip rumah sakit (gambar) dan/atau audio slip (suara), baca teks slip dengan OCR dan transkripsi suara.
    """
    result = {}

    # Proses gambar slip dengan OCR
    if foto_slip is not None:
        image_bytes = await foto_slip.read()
        slip_text = extract_slip_text(image_bytes)
        result["slip_text"] = slip_text

    # Proses audio slip dengan OpenAI Whisper
    if audio_slip is not None:
        audio_path = f"/tmp/{audio_slip.filename}"
        with open(audio_path, "wb") as f:
            f.write(await audio_slip.read())
        try:
            model = whisper.load_model("base")
            transcribe_result = model.transcribe(audio_path, language="indonesian")
            result["slip_audio_text"] = transcribe_result["text"]
        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)

    if not result:
        raise HTTPException(status_code=400, detail="Harus upload foto slip atau audio slip.")

    return result

@app.get("/")
async def root():
    return {
        "message": "Welcome to BISAcare - AI-Powered Insurance Assistant",
        "features": {
            "chatbot": "BISAbot dengan RAG terintegrasi untuk jawaban akurat",
            "document_analysis": "Analisis keluhan kesehatan dengan AI",
            "document_generation": "Generate surat aju banding otomatis",
            "voice_support": "Support input suara untuk analisis keluhan"
        },
        "endpoints": {
            "bisabot": "/bisabot (POST) - Chat dengan BISAbot (RAG terintegrasi)", # OK
            "bisabot_history": "/bisabot/history (GET) - Lihat riwayat chat", #OK
            "clear_history": "/bisabot/history (DELETE) - Hapus riwayat chat", #OK
            "surat_banding": "/surat_aju_banding (POST) - Buat surat aju banding",
            "analisis_keluhan": "/keluhanmu_bisa_diklaim (POST) - Analisis keluhan kesehatan (text)", #OK
            "download": "/download/{filename} (GET) - Download file PDF",
            "isi_data": "/isi_data (POST) - Upload foto KTP & Polis, dan data form lain",
            "bantu_proses_ai": "/bantu_proses_ai (POST) - Cek data isi_data dan saran AI",
            "upload_slip": "/slip_rumah_sakit (POST) - Upload slip rumah sakit dan ekstrak data",
            "get_slip": "/slip_rumah_sakit/{slip_id} (GET) - Ambil data slip rumah sakit berdasarkan ID",
            "get_keluhan": "/keluhanmu_bisa_diklaim/{keluhan_id} (GET) - Ambil data keluhan berdasarkan ID",
            "rekomendasi_rumah_sakit": "/rekomendasi_rumah_sakit (POST) - Rekomendasi rumah sakit", #OK
            "rekomendasi_asuransi": "/rekomendasi_asuransi (POST) - Rekomendasi asuransi" # OK
        },
        "setup": {
            "rag_documents": "Letakkan file PDF asuransi di folder: ./rag/documents/",
            "supported_audio": ["mp3", "wav", "m4a", "flac", "ogg", "webm", "mp4"]
        }
    }