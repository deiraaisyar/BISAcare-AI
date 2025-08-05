from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from features.bisabot.bisabot import ask_bisabot, get_chat_history, clear_chat_history
from features.surat_aju_banding.surat_aju_banding import buat_surat_aju_banding_pdf
import os
import uuid

app = FastAPI(title="BISAcare")

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

@app.post("/bisabot")
async def chat(query: Query):
    response = ask_bisabot(query.question)
    return {"answer": response}

@app.get("/bisabot/history")
async def get_history():
    history = get_chat_history()
    return {"history": history}

@app.delete("/bisabot/history")
async def clear_history():
    clear_chat_history()
    return {"message": "Chat history cleared successfully"}

@app.post("/surat_aju_banding")
async def buat_surat_banding(request: SuratAjuBandingRequest):
    try:
        # Generate unique filename
        unique_id = str(uuid.uuid4())[:8]
        nama_file = f"surat_aju_banding_{unique_id}.pdf"
        
        # Create the PDF
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
        
        # Check if file was created successfully
        if not os.path.exists(nama_file):
            raise HTTPException(status_code=500, detail="Gagal membuat file PDF")
        
        return {
            "message": "Surat aju banding berhasil dibuat",
            "filename": nama_file,
            "download_url": f"/download/{nama_file}"
        }
        
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
        "message": "Welcome to BISAcare API",
        "endpoints": {
            "bisabot": "/bisabot (POST) - Chat dengan BISAbot",
            "bisabot_history": "/bisabot/history (GET) - Lihat riwayat chat",
            "clear_history": "/bisabot/history (DELETE) - Hapus riwayat chat",
            "surat_banding": "/surat_aju_banding (POST) - Buat surat aju banding",
            "download": "/download/{filename} (GET) - Download file PDF"
        }
    }
