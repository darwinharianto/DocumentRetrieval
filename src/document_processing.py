from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)


def load_document(file_path, file_type="pdf"):
    """Loads a document from a file path."""
    if file_type == "pdf":
        return PyPDFLoader(file_path).load()
    raise ValueError(f"Unsupported file type: {file_type}")


def split_document(document, chunk_size=1000, chunk_overlap=100):
    """Splits a document into chunks using RecursiveCharacterTextSplitter."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],  # paragraph → line → sentence → word
    )
    return text_splitter.split_documents(document)
