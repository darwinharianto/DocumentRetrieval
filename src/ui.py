import streamlit as st
import os
import json
import shutil

from .config import FAISS_STORE_PATH, PROCESSED_FILES_LOG
from .document_processing import load_document, split_document
from .llm import get_embeddings_model, get_or_create_vectorstore, load_processed_files
from .retriever import RetrievalManager
from langchain_community.vectorstores import FAISS


def handle_file_uploads(uploaded_files, vectorstore, processed_files):
    """Processes uploaded files, creates embeddings, and updates the vector store."""
    new_files_processed = False
    embeddings = get_embeddings_model()

    for uploaded_file in uploaded_files:
        if uploaded_file.name not in processed_files:
            st.sidebar.info(f"Processing {uploaded_file.name}...")
            # Save the uploaded file temporarily
            with open(uploaded_file.name, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Load, split, and add to vectorstore
            document = load_document(uploaded_file.name, "pdf")
            texts = split_document(document)

            if vectorstore is None:
                vectorstore = FAISS.from_documents(texts, embeddings)
            else:
                vectorstore.add_documents(texts)

            processed_files.append(uploaded_file.name)
            os.remove(uploaded_file.name)  # Clean up temporary file
            new_files_processed = True

    if new_files_processed:
        st.sidebar.success("New files processed and added to the vector store.")
        vectorstore.save_local(FAISS_STORE_PATH)
        with open(PROCESSED_FILES_LOG, "w") as f:
            json.dump(processed_files, f)

    return vectorstore, processed_files


def main_ui():
    """Renders the main Streamlit UI."""
    st.title("Document Retrieval with Gemini API")
    st.sidebar.header("Settings")

    if st.sidebar.button("Clear Vector Store"):
        if os.path.exists(FAISS_STORE_PATH):
            shutil.rmtree(FAISS_STORE_PATH)
        if os.path.exists(PROCESSED_FILES_LOG):
            os.remove(PROCESSED_FILES_LOG)
        st.sidebar.success("Vector store cleared.")
        st.rerun()

    embeddings = get_embeddings_model()
    vectorstore = get_or_create_vectorstore(embeddings)
    processed_files = load_processed_files()

    uploaded_files = st.sidebar.file_uploader(
        "Upload PDF documents", type=["pdf"], accept_multiple_files=True
    )

    if uploaded_files:
        vectorstore, processed_files = handle_file_uploads(
            uploaded_files, vectorstore, processed_files
        )

    st.subheader("Uploaded Documents")
    if processed_files:
        for file_name in processed_files:
            st.write(f"- {file_name}")
    else:
        st.info("No documents have been uploaded and processed yet.")

    selected_files = []
    if processed_files:
        st.sidebar.subheader("Search Filter")
        selected_files = st.sidebar.multiselect(
            "Select files to search in:", options=processed_files, default=processed_files
        )

    if vectorstore is not None:
        retrieval_manager = RetrievalManager(vectorstore)
        retrieval_manager.setup_ensemble_retriever(selected_files)
        retrieval_chain = retrieval_manager.create_retrieval_chain()

        query = st.text_input("Enter your query:")
        if query:
            if retrieval_chain is None:
                st.error("Retrieval system not ready. Please upload documents.")
                return

            with st.spinner("Searching for answers..."):
                response = retrieval_chain.invoke({"input": query})
                st.subheader("Answer:")
                st.write(response.get("answer"))

                if docs := response.get("context", []):
                    st.subheader("Sources:")
                    for doc in docs:
                        source_name = doc.metadata.get("source", "Unknown")
                        page_number = doc.metadata.get("page")

                        display_page = f", Page {page_number + 1}" if page_number is not None else ""
                        
                        with st.expander(f"Source: {source_name}{display_page}"):
                            st.write(doc.page_content)
