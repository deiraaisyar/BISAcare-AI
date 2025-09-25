from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from datetime import datetime

def buat_surat_aju_banding_pdf(
    nama, no_polis, alamat, no_telepon,
    tanggal_pengajuan, nomor_klaim, perihal_klaim, alasan_penolakan, alasan_banding,
    nama_perusahaan_asuransi,
    nama_file_output="surat_aju_banding.pdf"
):
    doc = SimpleDocTemplate(
        nama_file_output,
        pagesize=A4,
        rightMargin=3*cm, leftMargin=3*cm,
        topMargin=2*cm, bottomMargin=2*cm
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Justify', alignment=4, leading=16))

    elements = []

    # Judul
    elements.append(Paragraph("<b>SURAT PENGAJUAN BANDING</b>", styles['Title']))
    elements.append(Spacer(1, 20))

    # Penerima
    elements.append(Paragraph(f"Kepada Yth.<br/>{nama_perusahaan_asuransi},<br/>Di tempat", styles['Normal']))
    elements.append(Spacer(1, 12))

    # Salam pembuka
    elements.append(Paragraph("Dengan hormat,", styles['Normal']))
    elements.append(Spacer(1, 12))

    # Identitas pengaju
    elements.append(Paragraph("Saya yang bertanda tangan di bawah ini:", styles['Normal']))
    elements.append(Spacer(1, 6))
    elements.append(Paragraph(
        f"Nama: {nama}<br/>"
        f"No. Polis: {no_polis}<br/>"
        f"Alamat: {alamat}<br/>"
        f"No. Telepon: {no_telepon}",
        styles['Normal']
    ))
    elements.append(Spacer(1, 12))

    # Isi surat
    elements.append(Paragraph(
        f"Dengan ini mengajukan banding atas penolakan klaim asuransi yang saya ajukan pada tanggal {tanggal_pengajuan}, "
        f"dengan nomor klaim {nomor_klaim} terkait {perihal_klaim}.",
        styles['Justify']
    ))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph(
        f"Kami menerima surat pemberitahuan penolakan dari pihak {nama_perusahaan_asuransi}, "
        f"dengan alasan penolakan sebagai berikut:",
        styles['Justify']
    ))
    elements.append(Paragraph(f"{alasan_penolakan}", styles['Justify']))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph(
        "Setelah mempertimbangkan kembali, saya merasa alasan penolakan tersebut tidak relevan dan sepihak. "
        "Berikut alasan banding saya:",
        styles['Justify']
    ))
    elements.append(Paragraph(f"{alasan_banding}", styles['Justify']))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph(
        "Sebagai bentuk tindak lanjut, saya melampirkan dokumen pendukung sebagai berikut:",
        styles['Justify']
    ))
    elements.append(Paragraph(
        "1. Fotokopi polis asuransi<br/>"
        "2. Surat penolakan klaim dari pihak asuransi<br/>"
        "3. Dokumen pendukung lainnya",
        styles['Normal']
    ))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph(
        f"Saya mohon pihak {nama_perusahaan_asuransi} dapat meninjau kembali keputusan tersebut "
        f"dengan pertimbangan yang lebih mendalam dan objektif.",
        styles['Justify']
    ))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph(
        "Demikian surat banding ini saya buat dengan harapan mendapat perhatian dan tanggapan yang bijak. "
        "Atas perhatian dan kerjasamanya, saya ucapkan terima kasih.",
        styles['Justify']
    ))
    elements.append(Spacer(1, 24))

    # Penutup
    elements.append(Paragraph("Hormat saya,", styles['Normal']))
    elements.append(Spacer(1, 48))
    elements.append(Paragraph(
        f"<b>{nama}</b><br/>{datetime.now().strftime('%-d %B %Y')}",
        styles['Normal']
    ))

    doc.build(elements)
    return nama_file_output
