import os
from typing import List
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_pdfs_from_folder(folder_path: str) -> List[str]:
    """
    Loads and splits text from all PDFs in a folder.
    
    Args:
        folder_path (str): Path to folder containing PDF files.
    
    Returns:
        List[str]: A list of text chunks from all PDFs.
    """
    documents = []
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, file)
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            
            # Split into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = text_splitter.split_documents(docs)
            documents.extend(chunks)
    
    return documents
