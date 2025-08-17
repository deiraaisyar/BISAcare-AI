import os
import numpy as np
import json
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm
from dotenv import load_dotenv

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from daftar_rumah_sakit.preprocessing import preprocessing_id

from PyPDF2 import PdfReader  # pastikan sudah install: pip install PyPDF2

def normalize(vectors):
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

def extract_pdf_text(pdf_path):
    """Ekstrak seluruh isi PDF sebagai satu string."""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text.strip()
    except Exception:
        return ""

def load_asuransi_json(folder_path):
    data = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            raw_text = extract_pdf_text(pdf_path)
            processed_text = preprocessing_id(raw_text)
            data.append({
                "nama_produk_asuransi": None,
                "nama_pt_asuransi": None,
                "contact_center_asuransi": None,
                "text": processed_text
            })
    return data

def save_json(data, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def generate_embeddings(texts, model):
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)
    return embeddings

def build_model(model_path=None):
    if model_path and os.path.exists(model_path):
        return SentenceTransformer(model_path)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    if model_path:
        model.save(model_path)
    return model

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def save_faiss_index(index, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    faiss.write_index(index, output_path)

def process_asuransi_data(folder_path, output_json, output_index, model_path=None):
    data = load_asuransi_json(folder_path)
    save_json(data, output_json)
    text_list = [d['text'] for d in data]
    model = build_model(model_path)
    embeddings = generate_embeddings(text_list, model)
    embeddings = normalize(embeddings)
    index = build_faiss_index(embeddings)
    save_faiss_index(index, output_index)
    print(f"Index saved to {output_index}")
    return data, index

if __name__ == "__main__":
    folder_path = "data"
    output_json = "preprocessed/daftar_asuransi_all.json"
    output_index = "app/embeddings/asuransi_st.index"
    model_path = "app/models/st_model"
    data, index = process_asuransi_data(folder_path, output_json, output_index, model_path)
    print("Asuransi model and index loaded.")