from fastapi import FastAPI, Request, HTTPException, File, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from features.bisabot.bisabot import ask_bisabot, get_chat_history, clear_chat_history, get_rag_status
from features.surat_aju_banding.surat_aju_banding import buat_surat_aju_banding_pdf
from features.keluhanmu_bisa_diklaim.keluhanmu_bisa_diklaim import analyze_health_complaint, analyze_health_complaint_from_audio
from features.hospital_recommender.hospital_recommender import recommend_hospitals
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
    umur: int
    jenis_layanan: str
    keluhan: str
    nama_asuransi: str
    nama_provinsi: str
    nama_daerah: str
    top_n: int = 5

# ============= BISABOT ENDPOINTS (with integrated RAG) =============

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

# ============= EXISTING ENDPOINTS =============

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

@app.post("/keluhanmu_bisa_diklaim")
async def analisis_keluhan(request: KeluhanRequest):
    try:
        if not request.keluhan_text.strip():
            raise HTTPException(status_code=400, detail="Keluhan tidak boleh kosong")
        
        result = analyze_health_complaint(request.keluhan_text)
        
        return {
            "status": "success",
            "keluhan_input": request.keluhan_text,
            "metode_input": request.metode_input,
            "analisis": {
                "persentase_kemungkinan_klaim": result.get("persentase_klaim", "Tidak dapat ditentukan"),
                "kemungkinan_diagnosis": result.get("kemungkinan_diagnosis", []),
                "rekomendasi_tindakan": result.get("rekomendasi_tindakan", []),
                "tingkat_urgensi": result.get("tingkat_urgensi", "sedang"),
                "dokumen_pendukung_diperlukan": result.get("dokumen_pendukung", [])
            },
            "disclaimer": "Hasil analisis ini hanya sebagai referensi. Konsultasikan dengan dokter untuk diagnosis yang akurat."
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error menganalisis keluhan: {str(e)}")

@app.post("/keluhanmu_bisa_diklaim/voice")
async def analisis_keluhan_voice(audio_file: UploadFile = File(...)):
    try:
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
            transcribed_text = result.get("transcribed_text", "")
            
            return {
                "status": "success",
                "keluhan_input": transcribed_text,
                "metode_input": "voice" if file_extension != '.mp4' else "video",
                "original_filename": audio_file.filename,
                "file_type": file_extension,
                "transcribed_text": transcribed_text,
                "analisis": {
                    "persentase_kemungkinan_klaim": result.get("persentase_klaim", "Tidak dapat ditentukan"),
                    "kemungkinan_diagnosis": result.get("kemungkinan_diagnosis", []),
                    "rekomendasi_tindakan": result.get("rekomendasi_tindakan", []),
                    "tingkat_urgensi": result.get("tingkat_urgensi", "sedang"),
                    "dokumen_pendukung_diperlukan": result.get("dokumen_pendukung", [])
                },
                "disclaimer": "Hasil analisis ini hanya sebagai referensi. Konsultasikan dengan dokter untuk diagnosis yang akurat."
            }
            
        finally:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
        
    except HTTPException:
        raise
    except Exception as e:
        try:
            if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
        except:
            pass
        raise HTTPException(status_code=500, detail=f"Error processing audio/video: {str(e)}")

# Load data, index, dan model sekali di awal
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
            "download": "/download/{filename} (GET) - Download file PDF"
        },
        "setup": {
            "rag_documents": "Letakkan file PDF asuransi di folder: ./rag/documents/",
            "supported_audio": ["mp3", "wav", "m4a", "flac", "ogg", "webm", "mp4"]
        }
    }