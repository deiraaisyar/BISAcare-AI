import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import os
import json
from tqdm import tqdm
from daftar_rumah_sakit.preprocessing import preprocessing_id

def normalize(vectors):
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

def load_json(input_path: str) -> list:
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def generate_embeddings(texts: list, model: SentenceTransformer) -> np.ndarray:
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)
    return embeddings

def build_model(model_path: str = None):
    if model_path and os.path.exists(model_path):
        return SentenceTransformer(model_path)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    if model_path:
        model.save(model_path)
    return model

def build_faiss_index(embeddings: np.ndarray):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def save_faiss_index(index, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    faiss.write_index(index, output_path)

def load_faiss_index(input_path: str):
    return faiss.read_index(input_path)

def process_hospital_data(input_path: str, output_path: str, model_path: str = None):
    data = load_json(input_path)
    text_list = [d['text'] for d in data]
    tqdm.pandas()
    text_list = [preprocessing_id(t) for t in tqdm(text_list)]
    if os.path.exists(output_path):
        index = load_faiss_index(output_path)
        print(f"Index loaded from {output_path}")
    else:
        model = build_model(model_path)
        embeddings = generate_embeddings(text_list, model)
        embeddings = normalize(embeddings)
        index = build_faiss_index(embeddings)
        save_faiss_index(index, output_path)
        print(f"Index saved to {output_path}")
    return data, index

if __name__ == "__main__":
    hospital_data, hospital_index = process_hospital_data(
        input_path="preprocessed/daftar_rumah_sakit_all.json",
        output_path="app/embeddings/hospital_st.index",
        model_path="app/models/st_model"
    )
    print("Hospital model and index loaded.")