from pydantic import BaseModel
from typing import List, Optional

class Query(BaseModel):
    question: str

class KeluhanRequest(BaseModel):
    text_keluhan: str

class KeluhanResponse(BaseModel):
    persentase_kondisi_dapat_diklaim: int
    kemungkinan_diagnosis: List[str]
    rekomendasi_tindakan: List[str]
    dokumen_pendukung_klaim: List[str]
    
class HospitalRequest(BaseModel):
    nama: str
    kelurahan_desa: str
    kecamatan: str
    jenis_layanan: str
    keluhan: str
    nama_asuransi: str
    nama_provinsi: str
    nama_daerah: str
    top_n: Optional[int] = 5

class HospitalItem(BaseModel):
    nama_rumah_sakit: str
    alamat: str
    telp: str
    text: str
    score: float

class HospitalResponse(BaseModel):
    results: List[HospitalItem]
    
class InsuranceRecommendRequest(BaseModel):
    query: str
    top_n: Optional[int] = 5

class InsuranceRecommendResponseItem(BaseModel):
    nama_produk_asuransi: str
    nama_pt_asuransi: str
    contact_center_asuransi: Optional[str]
    score: float

class KTPScanRequest(BaseModel):
    image_base64: str  # base64 string dari image KTP

class KTPScanResponse(BaseModel):
    nik: str | None
    nama: str | None
    tempat_lahir: str | None
    tanggal_lahir: str | None
    jenis_kelamin: str | None
    agama: str | None
    status_perkawinan: str | None
    pekerjaan: str | None
    kewarganegaraan: str | None
    kabupaten: str | None
    provinsi: str | None
    kelurahan_desa: str | None
    kecamatan: str | None
    raw: str | None

class AsuransiScanRequest(BaseModel):
    image_base64: str  # base64 string dari image KTP

class AsuransiScanResponse(BaseModel):
    coverage: str | None
    coverage_area: str | None
    jenis_asuransi: str | None
    nama: str | None
    nama_asuransi: str | None
    nomor_kartu: str | None
    nomor_polis: str | None
    raw: str | None

class InvoicersScanResponse(BaseModel):
    items: list[str] | None
    items_price: list[str] | None
    nama: str | None
    nama_rumah_sakit: str | None
    nomor_invoice: str | None
    tanggal: str | None
    total: str | None
    raw: str | None

class DiagnosisScanResponse(BaseModel):
    diagnosa: list[str] | None
    jenis_kelamin: str | None
    nama_dokter: str | None
    nama_pasien: str | None
    nik: str | None
    penyakit: str | None
    raw: str | None
    
    
class InsuranceGuideQuery(BaseModel):
    question: str

class CoverageDisplayRequest(BaseModel):
    diagnosis: dict
    asuransi: dict
    invoice: dict
    extra: Optional[dict] = None

class CoverageDisplayResponse(BaseModel):
    jenis_layanan: str
    deskripsi_layanan: str
    status_penanggungan: str
    persentasi_penanggungan: str
    limit_maksimum: str
    sisa_kuota: str
    estimasi_biaya_keluar: str
    alasan_status: str
    tanggal_efektif_penanggungan: str
    catatan_tambahan: str

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
    nama_perusahaan_asuransi: Optional[str] = None  # Diisi dari hasil scan asuransi

class SuratAjuBandingResponse(BaseModel):
    message: str
    filename: str
    download_url: str
    
class ClaimDenialRequest(BaseModel):
    user_message: str
    claim_data: dict
    
class DiagnosisTextRequest(BaseModel):
    diagnosis_text: str