from fastapi import FastAPI, Request, HTTPException, File, UploadFile, Form, Body
from fastapi.responses import FileResponse
from pydantic import BaseModel
from features.bisabot.bisabot import ask_bisabot, get_chat_history, clear_chat_history, get_rag_status
from features.surat_aju_banding.surat_aju_banding import buat_surat_aju_banding_pdf
from features.keluhanmu_bisa_diklaim.keluhanmu_bisa_diklaim import analyze_health_complaint, analyze_health_complaint_from_audio
from features.hospital_recommender.hospital_recommender import recommend_hospitals
from features.data_asuransi_ai.scan_data import extract_text, parse_with_ai
from features.bantu_proses_ai.bantu_proses_ai import cek_data_isi_data
from features.slip_rumah_sakit.slip_rumah_sakit import extract_text, parse_slip_with_ai
from features.insurance_recommender.insurance_recommender import load_asuransi_data, recommend_asuransi
from daftar_rumah_sakit.data_processing import load_faiss_index, load_json, build_model
import os
import uuid
import tempfile
import shutil
from typing import Optional

app = FastAPI(title="BISAcare - AI-Powered Insurance Assistant")

class Query(BaseModel):
    question: str

class SuratAjuBandingRequest(BaseModel):
    nama: str
    no_polis: str
    alamat: str
    no_telepon: str
    tanggal_pengajuan: str
    nomor_klaim: str
    perihal_klaim: str
    alasan_penolakan: str
    alasan_banding: str
    nama_perusahaan_asuransi: str

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

@app.post("/bisabot")
async def chat(query: Query):
    """Chat dengan BISAbot yang sudah terintegrasi dengan RAG"""
    response = ask_bisabot(query.question)
    return {"answer": response}

@app.get("/bisabot/history")
async def get_history():
    """Get chat history"""
    history = get_chat_history()
    return {"history": history}

@app.delete("/bisabot/history")
async def clear_history():
    """Clear chat history"""
    clear_chat_history()
    return {"message": "Chat history cleared successfully"}

@app.get("/bisabot/status")
async def get_status():
    """Get BISAbot and RAG status"""
    rag_status = get_rag_status()
    return {
        "bisabot": "online",
        "rag": rag_status
    }

@app.post("/surat_aju_banding")
async def buat_surat_banding(request: SuratAjuBandingRequest):
    try:
        unique_id = str(uuid.uuid4())[:8]
        nama_file = f"surat_aju_banding_{unique_id}.pdf"
        
        buat_surat_aju_banding_pdf(
            nama=request.nama,
            no_polis=request.no_polis,
            alamat=request.alamat,
            no_telepon=request.no_telepon,
            tanggal_pengajuan=request.tanggal_pengajuan,
            nomor_klaim=request.nomor_klaim,
            perihal_klaim=request.perihal_klaim,
            alasan_penolakan=request.alasan_penolakan,
            alasan_banding=request.alasan_banding,
            nama_perusahaan_asuransi=request.nama_perusahaan_asuransi,
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

@app.post("/rekomendasi_rumah_sakit")
async def rekomendasi_rumah_sakit(request: HospitalRecommendRequest):
    try:
        results = recommend_hospitals(
            data=hospital_data,
            index=hospital_index,
            model=hospital_model,
            nama=request.nama,
            kelurahan_desa=request.kelurahan_desa,
            kecamatan=request.kecamatan,
            umur=request.umur,
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

@app.post("/rekomendasi_asuransi")
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

@app.get("/download/{filename}")
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
    pilih_layanan: str = Form(...),
    nomor_hp: str = Form(...),
    input_keluhan: str = Form(...)
):
    """
    Upload foto KTP & Polis, plus data form lain.
    Output: hasil OCR & parsing + data form.
    """
    # OCR & AI parsing KTP
    ktp_bytes = await foto_ktp.read()
    ktp_raw_text = extract_text(ktp_bytes)
    ktp_parsed = parse_with_ai(ktp_raw_text)

    # OCR & AI parsing Polis
    polis_bytes = await foto_polis.read()
    polis_raw_text = extract_text(polis_bytes)
    polis_parsed = parse_with_ai(polis_raw_text)

    # Gabungkan hasil parsing
    result = {
        "ktp": ktp_parsed.get("ktp") if hasattr(ktp_parsed, "get") else ktp_parsed,
        "polis": polis_parsed.get("polis") if hasattr(polis_parsed, "get") else polis_parsed,
        "raw_text": f"{ktp_raw_text}\n{polis_raw_text}",
        "nomor_polis": nomor_polis,
        "layanan": pilih_layanan,
        "nomor_hp": nomor_hp,
        "keluhan": input_keluhan
    }
    return result

@app.post("/bantu_proses_ai")
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

@app.post("/keluhanmu_bisa_diklaim")
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
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                shutil.copyfileobj(audio_file.file, temp_file)
                temp_file_path = temp_file.name
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

@app.get("/keluhanmu_bisa_diklaim/{keluhan_id}")
async def get_keluhanmu_bisa_diklaim(keluhan_id: str):
    data = keluhan_data_store.get(keluhan_id)
    if not data:
        raise HTTPException(status_code=404, detail="Data keluhan tidak ditemukan")
    return data

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
            "bisabot": "/bisabot (POST) - Chat dengan BISAbot (RAG terintegrasi)",
            "bisabot_history": "/bisabot/history (GET) - Lihat riwayat chat",
            "bisabot_status": "/bisabot/status (GET) - Status BISAbot dan RAG",
            "clear_history": "/bisabot/history (DELETE) - Hapus riwayat chat",
            "surat_banding": "/surat_aju_banding (POST) - Buat surat aju banding",
            "analisis_keluhan": "/keluhanmu_bisa_diklaim (POST) - Analisis keluhan kesehatan (text)",
            "analisis_keluhan_voice": "/keluhanmu_bisa_diklaim/voice (POST) - Analisis keluhan dari audio/video",
            "download": "/download/{filename} (GET) - Download file PDF",
            "isi_data": "/isi_data (POST) - Upload foto KTP & Polis, dan data form lain",
            "bantu_proses_ai": "/bantu_proses_ai (POST) - Cek data isi_data dan saran AI",
            "upload_slip": "/slip_rumah_sakit (POST) - Upload slip rumah sakit dan ekstrak data",
            "get_slip": "/slip_rumah_sakit/{slip_id} (GET) - Ambil data slip rumah sakit berdasarkan ID",
            "get_keluhan": "/keluhanmu_bisa_diklaim/{keluhan_id} (GET) - Ambil data keluhan berdasarkan ID",
            "rekomendasi_rumah_sakit": "/rekomendasi_rumah_sakit (POST) - Rekomendasi rumah sakit",
            "rekomendasi_asuransi": "/rekomendasi_asuransi (POST) - Rekomendasi asuransi"
        },
        "setup": {
            "rag_documents": "Letakkan file PDF asuransi di folder: ./rag/documents/",
            "supported_audio": ["mp3", "wav", "m4a", "flac", "ogg", "webm", "mp4"]
        }
    }