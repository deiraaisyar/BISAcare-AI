import numpy as np
from daftar_rumah_sakit.preprocessing import preprocessing_id
import os
import json

def recommend_asuransi(
    query, data, index, model, top_n=5
) -> list:
    """
    Merekomendasikan produk asuransi berdasarkan input user dan kemiripan embedding.
    """
    # Preprocessing query
    query_text = preprocessing_id(query)
    query_emb = model.encode([query_text])
    query_emb = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)

    # Cari kemiripan di index
    D, I = index.search(query_emb, top_n)
    results = []
    for idx, dist in zip(I[0], D[0]):
        d = data[idx]
        results.append({
            'nama_produk_asuransi': d.get('nama_produk_asuransi', ''),
            'nama_pt_asuransi': d.get('nama_pt_asuransi', ''),
            'contact_center_asuransi': d.get('contact_center_asuransi', None),
            'score': float(dist)
        })
    return results

def load_asuransi_data(folder_path):
    data_path = os.path.join(folder_path, "preprocessed/daftar_asuransi_all.json")
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data