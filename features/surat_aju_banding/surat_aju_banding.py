from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from datetime import datetime

def buat_surat_aju_banding_pdf(
    nama, no_polis, alamat, no_telepon,
    tanggal_pengajuan, nomor_klaim, perihal_klaim, alasan_penolakan, alasan_banding,
    nama_perusahaan_asuransi="PT ASURANSI SINARMAS",
    nama_file_output="surat_aju_banding.pdf"
):
    doc = SimpleDocTemplate(nama_file_output, pagesize=A4,
                            rightMargin=3*cm, leftMargin=3*cm,
                            topMargin=2*cm, bottomMargin=2*cm)

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Justify', alignment=4, leading=16))

    elements = []

    # Header
    elements.append(Paragraph(f"<b>SURAT PENGAJUAN BANDING</b>", styles['Title']))
    elements.append(Spacer(1, 20))

    elements.append(Paragraph(f"Kepada Yth.<br/>{nama_perusahaan_asuransi},<br/>Di tempat", styles['Normal']))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("Dengan hormat,", styles['Normal']))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("Saya yang bertanda tangan di bawah ini:", styles['Normal']))
    elements.append(Spacer(1, 6))

    elements.append(Paragraph(f"Nama: {nama}<br/>No. Polis: {no_polis}<br/>Alamat: {alamat}<br/>No. Telepon: {no_telepon}", styles['Normal']))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph(f"Dengan ini mengajukan banding atas penolakan klaim asuransi yang saya ajukan pada tanggal {tanggal_pengajuan}, dengan nomor klaim {nomor_klaim} terkait {perihal_klaim}.", styles['Justify']))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph(f"Kami menerima surat pemberitahuan penolakan dari pihak {nama_perusahaan_asuransi}, dengan alasan penolakan sebagai berikut:", styles['Justify']))
    elements.append(Paragraph(f"{alasan_penolakan}", styles['Justify']))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph(f"Setelah mempertimbangkan kembali, saya merasa alasan penolakan tersebut tidak relevan dan sepihak. Berikut alasan banding saya:", styles['Justify']))
    elements.append(Paragraph(f"{alasan_banding}", styles['Justify']))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("Sebagai bentuk tindak lanjut, saya melampirkan dokumen pendukung sebagai berikut:", styles['Justify']))
    elements.append(Paragraph("1. Fotokopi polis asuransi<br/>2. Surat penolakan klaim dari pihak asuransi<br/>3. Dokumen pendukung lainnya", styles['Normal']))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph(f"Saya mohon pihak {nama_perusahaan_asuransi} dapat meninjau kembali keputusan tersebut dengan pertimbangan yang lebih mendalam dan objektif.", styles['Justify']))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("Demikian surat banding ini saya buat dengan harapan mendapat perhatian dan tanggapan yang bijak. Atas perhatian dan kerjasamanya, saya ucapkan terima kasih.", styles['Justify']))
    elements.append(Spacer(1, 24))

    elements.append(Paragraph("Hormat saya,", styles['Normal']))
    elements.append(Spacer(1, 48))

    elements.append(Paragraph(f"<b>{nama}</b><br/>{datetime.now().strftime('%-d %B %Y')}", styles['Normal']))

    # Build PDF
    doc.build(elements)
    print(f"Surat berhasil dibuat: {nama_file_output}")

# Contoh penggunaan:
if __name__ == "__main__":
    buat_surat_aju_banding_pdf(
        nama="Rina Vina Wijaya",
        no_polis="0208202400316",
        alamat="Permata Raya Sukodono e4-18, Sidoarjo",
        no_telepon="0812-8999-9628",
        tanggal_pengajuan="26 Mei 2025",
        nomor_klaim="007/KLAIM-ASM SBY/7/2025",
        perihal_klaim="kerusakan kendaraan yaris 2019 dengan nopol L 1637 ZA",
        alasan_penolakan="1. Estimasi biaya yang hanya mencapai 72%<br/>2. Keterlambatan pajak",
        alasan_banding="Penolakan dirasa sepihak karena tidak menjelaskan kerusakan dan pajak secara rinci. Keterlambatan pajak disebabkan oleh status BPKB yang berada di oto finance, bukan kelalaian pribadi."
    )
