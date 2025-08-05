import os
import logging
from typing import List, Dict
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

logger = logging.getLogger(__name__)

class DocumentLoader:
    def __init__(self, documents_path: str = "./rag/documents"):
        self.documents_path = documents_path
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
        # Create documents directory if it doesn't exist
        os.makedirs(documents_path, exist_ok=True)
    
    def load_pdf(self, file_path: str) -> List[Document]:
        """Load and split a single PDF file"""
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            # Add simple metadata
            for doc in documents:
                doc.metadata.update({
                    "source": os.path.basename(file_path),
                    "file_type": "pdf"
                })
            
            # Split documents
            split_docs = self.text_splitter.split_documents(documents)
            logger.info(f"Loaded {len(split_docs)} chunks from {file_path}")
            
            return split_docs
            
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {str(e)}")
            return []
    
    def load_all_documents(self) -> List[Document]:
        """Load all PDF documents from the documents directory"""
        all_documents = []
        
        if not os.path.exists(self.documents_path):
            logger.warning(f"Documents path {self.documents_path} does not exist")
            return all_documents
        
        pdf_files = [f for f in os.listdir(self.documents_path) if f.endswith('.pdf')]
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {self.documents_path}")
            return all_documents
        
        for pdf_file in pdf_files:
            file_path = os.path.join(self.documents_path, pdf_file)
            documents = self.load_pdf(file_path)
            all_documents.extend(documents)
        
        logger.info(f"Total loaded documents: {len(all_documents)} chunks from {len(pdf_files)} PDF files")
        return all_documents

# Test loading
if __name__ == "__main__":
    loader = DocumentLoader()
    docs = loader.load_all_documents()
    print(f"Loaded {len(docs)} document chunks")
    if docs:
        print(f"Sample chunk: {docs[0].page_content[:200]}...")
        print(f"Metadata: {docs[0].metadata}")