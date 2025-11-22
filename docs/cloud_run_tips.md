# Cloud Run deployment tips & memory optimization

These notes help when deploying the FastAPI app to Cloud Run and encountering memory/time issues.

1) Increase memory when deploying

```bash
# deploy with 8Gi memory and 1 max instance as an example
gcloud run deploy bisacare-ai \
  --image gcr.io/<PROJECT_ID>/bisacare-ai:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 8Gi \
  --timeout 300s
```

2) Alternative: reduce runtime footprint

- Use a smaller embedding model for the recommender/RAG. Example replacements:
  - `paraphrase-MiniLM-L6-v2` or `all-MiniLM-L6-v2` instead of larger multilingual models.
- Lazy-load large models or the FAISS index: only load the index/embedding model on the first request.
- Move heavy computation into a separate background worker or server (e.g., Cloud Run job or Cloud Functions) and keep the API lightweight.

3) Rebuild FAISS index with a smaller embedding model (example python snippet)

```python
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
docs = [...]  # list of text documents
embs = model.encode(docs, show_progress_bar=True, convert_to_numpy=True)

dim = embs.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embs)
faiss.write_index(index, 'faiss_index_small.bin')

with open('metadata.pkl', 'wb') as f:
    pickle.dump(docs, f)
```

Push the index and metadata to a Hugging Face repo (git lfs required for .bin/.pkl files):

```bash
git lfs install
git lfs track "*.bin" "*.pkl"
git add .gitattributes faiss_index_small.bin metadata.pkl
git commit -m "Add small faiss index"
git push origin main
```

4) Useful debugging steps

- Check Cloud Run logs (Stackdriver) for memory OOM or timeouts.
- Try running the container locally with limited resources to reproduce: `docker run -m 6g ...`.
- Consider splitting CPU-heavy tasks into Cloud Run jobs or Cloud Tasks.
