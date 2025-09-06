import os
import streamlit as st
import google.generativeai as genai

# --- Constants ---
FAISS_STORE_PATH = "faiss_store"
PROCESSED_FILES_LOG = "processed_files.json"


def setup_api():
    """Set up the Google Generative AI API."""
    # Set up Google Generative AI API
    os.environ["GOOGLE_API_KEY"] = st.secrets["API_KEY"]
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
