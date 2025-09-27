from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
import tempfile
from features_fixed.bisabot import ask_bisabot, get_chat_history, clear_chat_history
from features_fixed.voice_transcribe import transcribe_audio
from schema_fixed.schemas import Query, KeluhanRequest, KeluhanResponse, HospitalRequest, HospitalResponse, HospitalItem, InsuranceRecommendRequest, KTPScanResponse, KTPScanRequest, AsuransiScanResponse, InvoicersScanResponse, DiagnosisScanResponse,InsuranceGuideQuery, CoverageDisplayRequest, CoverageDisplayResponse, SuratAjuBandingRequest, SuratAjuBandingResponse, ClaimDenialRequest, DiagnosisTextRequest
from features_fixed.cek_tanggapan import process_keluhan
from features_fixed.hospital_recommender import recommend_hospitals
from daftar_rumah_sakit.data_processing import load_faiss_index, load_json, build_model
from features_fixed.insurance_recommender import recommend_asuransi, load_asuransi_data
from features_fixed.scan_ktp import scan_ktp_pipeline
from features_fixed.scan_asuransi import scan_asuransi_pipeline
from features_fixed.scan_invoicers import scan_invoicers_pipeline
from features_fixed.scan_diagnosis import scan_diagnosis_pipeline
import base64
from pydantic import BaseModel
from features_fixed.ai_insurance_guide import ask_ai_insurance_guide, get_ai_insurance_guide_history, clear_ai_insurance_guide_history
from features_fixed.ai_coverage_display import coverage_ai_pipeline
from features_fixed.ai_claim_denial import claim_denial_chatbot
from features_fixed.generate_ajubanding import buat_surat_aju_banding_pdf
from datetime import datetime
from features_fixed.diagnosis_text import diagnosis_text_pipeline
from sentence_transformers import SentenceTransformer
import tempfile
from pydub import AudioSegment

app = FastAPI()

# Load Hugging Face model
hospital_model = SentenceTransformer("ayaayaa/hospital-recommender")
asuransi_model = SentenceTransformer("ayaayaa/insurance-recommender")

# Rumah sakit data dan model paths
DATA_PATH = "daftar_rumah_sakit/preprocessed/daftar_rumah_sakit_all.json"
INDEX_PATH = "daftar_rumah_sakit/app/embeddings/hospital_st.index"

hospital_data = load_json(DATA_PATH)
hospital_index = load_faiss_index(INDEX_PATH)

ASURANSI_DATA_PATH = "daftar_asuransi/preprocessed/daftar_asuransi_all.json"
ASURANSI_INDEX_PATH = "daftar_asuransi/app/embeddings/asuransi_st.index"

asuransi_data = load_json(ASURANSI_DATA_PATH)
asuransi_index = load_faiss_index(ASURANSI_INDEX_PATH)

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import tempfile
from pydub import AudioSegment

app = FastAPI()

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    try:
        # Simpan file original (m4a, wav, dll.)
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.filename.split('.')[-1]}") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # Konversi ke wav (jika bukan wav)
        if not tmp_path.endswith(".wav"):
            audio = AudioSegment.from_file(tmp_path)
            wav_path = tmp_path + ".wav"
            audio.export(wav_path, format="wav")
        else:
            wav_path = tmp_path

        # Panggil fungsi transkripsi
        transcription = transcribe_audio(wav_path)

        return JSONResponse({"text": transcription})
    
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/bisabot")
async def chat(query: Query):
    """Chat dengan BISAbot yang sudah terintegrasi dengan RAG"""
    response = ask_bisabot(query.question)
    return {"answer": response}

@app.get("/bisabot/history")
async def history():
    """Get chat history"""
    history = get_chat_history()
    return {"history": history}

@app.delete("/bisabot/history")
async def clear():
    """Clear chat history"""
    clear_chat_history()
    return {"message": "Chat history cleared successfully"}

@app.post("/cek-tanggapan", response_model=KeluhanResponse)
async def cek_tanggapan(req: KeluhanRequest):
    result = process_keluhan(req.text_keluhan)
    return JSONResponse(result)

@app.post("/rekomendasi-rumah-sakit", response_model=HospitalResponse)
async def rekomendasi_rumah_sakit(request: HospitalRequest):
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

        hospital_items = [HospitalItem(**r) for r in results]
        return HospitalResponse(results=hospital_items)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    
@app.post("/rekomendasi-asuransi")
async def rekomendasi_asuransi_endpoint(request: InsuranceRecommendRequest):
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

@app.post("/scan-ktp", response_model=KTPScanResponse)
async def scan_ktp(file: UploadFile = File(...)):
    try:
        # Baca file gambar dan encode ke base64
        image_bytes = await file.read()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        parsed_data = scan_ktp_pipeline(image_base64)
        return parsed_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/scan-asuransi", response_model=AsuransiScanResponse)
async def scan_asuransi(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        parsed_data = scan_asuransi_pipeline(image_base64)
        return parsed_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/scan-invoicers", response_model=InvoicersScanResponse)
async def scan_diagnosis_invoice(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        parsed_data = scan_invoicers_pipeline(image_base64)
        return parsed_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/scan-diagnosis", response_model=DiagnosisScanResponse)
async def scan_diagnosis(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        parsed_data = scan_diagnosis_pipeline(image_base64)
        return parsed_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/ai-insurance-guide")
async def ai_insurance_guide_chat(query: InsuranceGuideQuery):
    response = ask_ai_insurance_guide(query.question)
    return {"answer": response}

@app.get("/ai-insurance-guide/history")
async def ai_insurance_guide_history():
    return {"history": get_ai_insurance_guide_history()}

@app.delete("/ai-insurance-guide/history")
async def ai_insurance_guide_clear():
    clear_ai_insurance_guide_history()
    return {"message": "AI Insurance Guide chat history cleared successfully"}

@app.post("/ai-coverage-display", response_model=CoverageDisplayResponse)
async def ai_coverage_display(request: CoverageDisplayRequest):
    result = coverage_ai_pipeline(
        diagnosis=request.diagnosis,
        asuransi=request.asuransi,
        invoice=request.invoice,
        extra=request.extra
    )
    return result

@app.post("/ai-claim-denial")
async def ai_claim_denial(request: ClaimDenialRequest):
    # Isi required_fields sesuai field penting dari schemas.py
    required_fields = [
        "nik", "nama", "nomor_polis", "nomor_kartu", "dokumen_invoice", "dokumen_kwitansi", "dokumen_diagnosis"
    ]
    response = claim_denial_chatbot(request.user_message, request.claim_data, required_fields)
    return {"answer": response}

@app.post("/surat-aju-banding", response_model=SuratAjuBandingResponse)
async def surat_aju_banding(request: SuratAjuBandingRequest):
    # Jika nama_perusahaan_asuransi tidak diisi, ambil dari hasil scan asuransi
    nama_perusahaan = request.nama_perusahaan_asuransi
    if not nama_perusahaan:
        nama_perusahaan = "PT ASURANSI"  # fallback jika tidak ada data

    filename = f"surat_aju_banding_{request.nama}_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"
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
        nama_perusahaan_asuransi=nama_perusahaan,
        nama_file_output=filename
    )
    return SuratAjuBandingResponse(
        message="Surat aju banding berhasil dibuat",
        filename=filename,
        download_url=f"/download/{filename}"
    )

@app.get("/download/{filename}")
async def download_file(filename: str):
    return FileResponse(filename, media_type="application/pdf", filename=filename)

@app.post("/diagnosis-text", response_model=DiagnosisScanResponse)
async def diagnosis_text(request: DiagnosisTextRequest):
    result = diagnosis_text_pipeline(request.diagnosis_text)
    return result