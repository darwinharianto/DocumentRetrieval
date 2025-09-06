import os
import json
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from .config import FAISS_STORE_PATH, PROCESSED_FILES_LOG


def get_embeddings_model():
    """Returns the Google Generative AI Embeddings model."""
    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004", transport="rest"
    )


def get_or_create_vectorstore(embeddings):
    """Loads an existing vector store or creates a new one."""
    if os.path.exists(FAISS_STORE_PATH):
        st.sidebar.info("Loading existing vector store...")
        return FAISS.load_local(
            FAISS_STORE_PATH, embeddings, allow_dangerous_deserialization=True
        )
    st.sidebar.info("Creating a new vector store.")
    return None


def load_processed_files():
    """Loads the log of processed files."""
    if os.path.exists(PROCESSED_FILES_LOG):
        with open(PROCESSED_FILES_LOG, "r") as f:
            return json.load(f)
    return []
