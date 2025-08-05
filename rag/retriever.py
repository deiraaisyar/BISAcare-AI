import os
import logging
from typing import List, Dict, Any
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.schema import Document
from .loader import DocumentLoader
import pickle

logger = logging.getLogger(__name__)

class SimpleRAGRetriever:
    def __init__(self, 
                 documents_path: str = "./rag/documents",
                 embeddings_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                 index_path: str = "./rag/index"):
        
        self.documents_path = documents_path
        self.embeddings_model_name = embeddings_model
        self.index_path = index_path
        
        # Create index directory if it doesn't exist
        os.makedirs(index_path, exist_ok=True)
        
        # Initialize embedding model
        try:
            self.embeddings_model = SentenceTransformer(embeddings_model)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            self.embeddings_model = None
        
        # Initialize components
        self.documents: List[Document] = []
        self.index = None
        self.document_embeddings = None
        
        # Load documents and create index
        self._initialize()
    
    def _save_index(self):
        """Save FAISS index and metadata to disk"""
        try:
            if self.index is None or self.document_embeddings is None:
                logger.warning("No index to save")
                return
            
            # Save FAISS index
            index_file = os.path.join(self.index_path, "faiss_index.bin")
            faiss.write_index(self.index, index_file)
            
            # Save documents and embeddings metadata
            metadata_file = os.path.join(self.index_path, "metadata.pkl")
            metadata = {
                'documents': self.documents,
                'embeddings_model_name': self.embeddings_model_name,
                'document_count': len(self.documents)
            }
            
            with open(metadata_file, 'wb') as f:
                pickle.dump(metadata, f)
            
            logger.info(f"Index saved to {self.index_path}")
            
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}")
    
    def _load_index(self):
        """Load FAISS index and metadata from disk"""
        try:
            index_file = os.path.join(self.index_path, "faiss_index.bin")
            metadata_file = os.path.join(self.index_path, "metadata.pkl")
            
            if not os.path.exists(index_file) or not os.path.exists(metadata_file):
                logger.info("No saved index found")
                return False
            
            # Load FAISS index
            self.index = faiss.read_index(index_file)
            
            # Load metadata
            with open(metadata_file, 'rb') as f:
                metadata = pickle.load(f)
            
            self.documents = metadata['documents']
            
            # Check if model matches
            if metadata['embeddings_model_name'] != self.embeddings_model_name:
                logger.warning("Embeddings model mismatch, recreating index")
                return False
            
            logger.info(f"Loaded index with {len(self.documents)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            return False
    
    def _initialize(self):
        """Initialize the RAG system"""
        try:
            if not self.embeddings_model:
                logger.warning("Embedding model not available, RAG disabled")
                return
            
            # Try to load existing index first
            if self._load_index():
                logger.info("Loaded existing index")
                return
                
            # Load documents
            loader = DocumentLoader(self.documents_path)
            self.documents = loader.load_all_documents()
            
            if not self.documents:
                logger.warning("No documents found, RAG will not work")
                return
            
            # Create embeddings and index
            self._create_index()
            
            # Save the index for future use
            self._save_index()
            
        except Exception as e:
            logger.error(f"Error initializing RAG: {str(e)}")
    
    def _create_index(self):
        """Create FAISS index from documents"""
        try:
            logger.info("Creating embeddings for documents...")
            
            # FIX: Properly format texts for encoding
            texts = []
            for doc in self.documents:
                text_content = doc.page_content
                # Ensure text is a string and clean it
                if isinstance(text_content, str):
                    # Clean the text
                    cleaned_text = text_content.strip()
                    if cleaned_text:  # Only add non-empty texts
                        texts.append(cleaned_text)
                else:
                    logger.warning(f"Skipping non-string content: {type(text_content)}")
            
            if not texts:
                logger.error("No valid texts to encode")
                return
            
            logger.info(f"Encoding {len(texts)} text chunks...")
            
            # FIX: Encode with proper error handling
            try:
                self.document_embeddings = self.embeddings_model.encode(
                    texts,  # Pass list of strings directly
                    show_progress_bar=True,
                    batch_size=16,  # Reduce batch size for stability
                    convert_to_numpy=True,  # Ensure numpy output
                    normalize_embeddings=False  # We'll normalize manually
                )
                
                logger.info(f"Successfully created embeddings with shape: {self.document_embeddings.shape}")
                
            except Exception as e:
                logger.error(f"Error during encoding: {str(e)}")
                # Try with smaller batch size
                logger.info("Retrying with batch size 1...")
                embeddings_list = []
                for i, text in enumerate(texts):
                    try:
                        emb = self.embeddings_model.encode([text])
                        embeddings_list.append(emb[0])
                        if i % 50 == 0:
                            logger.info(f"Processed {i+1}/{len(texts)} texts")
                    except Exception as text_error:
                        logger.warning(f"Skipping problematic text at index {i}: {str(text_error)}")
                        continue
                
                if embeddings_list:
                    self.document_embeddings = np.array(embeddings_list)
                    # Update documents list to match embeddings
                    self.documents = [self.documents[i] for i in range(len(embeddings_list))]
                else:
                    logger.error("Failed to create any embeddings")
                    return
            
            # Create FAISS index
            dimension = self.document_embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(self.document_embeddings)
            self.index.add(self.document_embeddings.astype('float32'))
            
            logger.info(f"Created FAISS index with {len(self.documents)} documents")
            
        except Exception as e:
            logger.error(f"Error creating index: {str(e)}")
            self.index = None
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for a query"""
        if not self.index or not self.documents or not self.embeddings_model:
            logger.warning("RAG system not properly initialized")
            return []
        
        try:
            # FIX: Properly encode query
            if not isinstance(query, str) or not query.strip():
                logger.warning("Invalid query provided")
                return []
            
            # Encode query
            query_embedding = self.embeddings_model.encode([query.strip()])
            
            # Ensure proper shape and type
            if len(query_embedding.shape) == 2:
                query_embedding = query_embedding[0]  # Take first embedding if batch
            
            query_embedding = query_embedding.reshape(1, -1)  # Ensure 2D shape
            faiss.normalize_L2(query_embedding)
            
            # Search
            scores, indices = self.index.search(
                query_embedding.astype('float32'), 
                min(top_k, len(self.documents))
            )
            
            # Format results
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx != -1 and score > 0.1:  # Filter low-quality matches
                    doc = self.documents[idx]
                    results.append({
                        'content': doc.page_content,
                        'metadata': doc.metadata,
                        'score': float(score),
                        'rank': i + 1
                    })
            
            logger.info(f"Retrieved {len(results)} relevant documents for query")
            return results
            
        except Exception as e:
            logger.error(f"Error during retrieval: {str(e)}")
            return []
    
    def get_context_for_query(self, query: str, max_context_length: int = 1500) -> str:
        """Get formatted context for RAG (simplified)"""
        relevant_docs = self.retrieve(query, top_k=3)
        
        if not relevant_docs:
            return ""
        
        context_parts = []
        current_length = 0
        
        for doc in relevant_docs:
            content = doc['content'].strip()
            source = doc['metadata'].get('source', 'Unknown')
            
            # Simple formatting without document type
            formatted_content = f"[Dari: {source}]\n{content}\n"
            
            if current_length + len(formatted_content) <= max_context_length:
                context_parts.append(formatted_content)
                current_length += len(formatted_content)
            else:
                break
        
        context = "\n".join(context_parts)
        logger.info(f"Generated context with {len(context)} characters from {len(context_parts)} documents")
        
        return context
    
    def is_available(self) -> bool:
        """Check if RAG system is available and working"""
        return (self.embeddings_model is not None and 
                self.index is not None and 
                len(self.documents) > 0)
    
    def refresh_index(self):
        """Refresh index by reloading documents and recreating embeddings"""
        try:
            logger.info("Refreshing RAG index...")
            
            # Load documents again
            loader = DocumentLoader(self.documents_path)
            self.documents = loader.load_all_documents()
            
            if not self.documents:
                logger.warning("No documents found after refresh")
                return False
            
            # Recreate index
            self._create_index()
            
            # Save new index
            self._save_index()
            
            logger.info("Index refreshed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error refreshing index: {str(e)}")
            return False