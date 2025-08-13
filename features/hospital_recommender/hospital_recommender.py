import numpy as np
from dotenv import load_dotenv
import os
from typing import Optional
from daftar_rumah_sakit.preprocessing import preprocessing_id
from daftar_rumah_sakit.data_processing import normalize

def recommend_hospitals(
    data, index, model,
    nama: str,
    kelurahan_desa: str,
    kecamatan: str,
    umur: int,
    jenis_layanan: str,
    keluhan: str,
    nama_asuransi: str,
    nama_provinsi: str,
    nama_daerah: str,
    top_n: int = 5
) -> list:
    """
    Merekomendasikan rumah sakit berdasarkan input user dan kemiripan embedding.
    """
    # Gabungkan semua input jadi satu query
    query_text = (
        f"{nama} {kelurahan_desa} {kecamatan} umur:{umur} layanan:{jenis_layanan} "
        f"keluhan:{keluhan} asuransi:{nama_asuransi} provinsi:{nama_provinsi} daerah:{nama_daerah}"
    )
    query_text = preprocessing_id(query_text)
    query_emb = model.encode([query_text])
    query_emb = normalize(query_emb)

    # Cari kemiripan di index
    D, I = index.search(query_emb, top_n)
    results = []
    for idx, dist in zip(I[0], D[0]):
        d = data[idx]
        results.append({
            'nama_rumah_sakit': d.get('nama_rumah_sakit', ''),
            'alamat': d.get('alamat', ''),
            'telp': d.get('telp', ''),
            'text': d.get('text', ''),
            'score': float(dist)
        })
    return results