# Document Retrieval and Summarization Tool
## Overview

This project is a prototype web application that helps users efficiently find and understand information from multiple documents.

Users can:

1. Ask natural language questions.
2. Retrieve the most relevant documents using vector search.
3. Summarize those documents with an LLM (Google Gemini API).

This addresses the real-world challenge of document fragmentation in organizations, where critical knowledge is distributed across many files.

----
Features

- 📂 Document Ingestion – Upload PDF documents for indexing.

- 🔍 Semantic Search – Retrieve relevant documents using vector embeddings.

- ✍️ LLM Summarization – Summarize and answer questions with Gemini.

- 🌐 Web Interface – Simple UI built with Streamlit.

- ⚡ Persistent Storage – FAISS vector store caches processed files for reuse.

- 🗑️ Reset Option – Clear the index and start fresh.


## Architecture

```
+------------------+
|   PDF Documents  |
+------------------+
          |
          v
+---------------------------+
| Document Loader (PyPDF)   |
| Text Splitter             |
+---------------------------+
          |
          v
+---------------------------+
| Embeddings (Gemini API)   |
+---------------------------+
          |
          v
+---------------------------+
| Vector Store (FAISS)      |
+---------------------------+
          |
          v
+---------------------------+
| Retrieval + Summarization |
| (Gemini LLM via LangChain)|
+---------------------------+
          |
          v
+---------------------------+
|   Streamlit Web UI        |
+---------------------------+
```

### Setup Prerequisites

- Python 3.10+

- Google Gemini API key (set in Streamlit secrets or .env)


### Installation
```
git clone https://github.com/darwinharianto/DocumentRetrieval
cd DocumentRetrieval
pip install -r requirements.txt
```

### Run
```
streamlit run main.py
```

### Configuration
In `.streamlit/secrets.toml` :
```
API_KEY = "your-google-gemini-api-key"
```

### Docker setup
This application is designed to be run inside a Docker container.
Prerequisites

- Docker: Ensure you have Docker installed and running on your system.
- Google API Key: Your API key for the Google Gemini API.

1. Build the Docker Image

Open your terminal, navigate to the project directory, and run the following command to build the Docker image. This process may take a few minutes as it downloads the necessary dependencies.

```
docker build -t document-retrieval-app .
```

2. Run the Docker Container

Once the image is built, you can run the container. You must pass your Google API key as an environment variable using the -e flag.

Replace YOUR_GOOGLE_API_KEY with your actual API key.

```
docker run -p 8501:8501 -e API_KEY="YOUR_GOOGLE_API_KEY" document-retrieval-app
```

3. Access the Application

After running the command, you should see a message in your terminal indicating that the application is running. You can then access the app by opening a web browser and navigating to:

```
http://localhost:8501
```

You can now upload PDF documents and ask questions based on their content.


### How It Works

1. Upload PDFs → Text is extracted, split into chunks.
2. Embed Chunks → Each chunk is converted into a vector using text-embedding-004.
3. Vector Store (FAISS) → Chunks are stored for efficient similarity search.
4. Query → User question is embedded and matched against stored vectors.
5. Summarization → Top documents are passed to Gemini LLM, which generates a concise answer + references.

#### Technology Choices & Rationale

1. FAISS – Lightweight, efficient local vector search; avoids complexity of external services (Pinecone, Elasticsearch) while sufficient for a 5-doc prototype.
2. Gemini API (LLM + Embeddings) – Multilingual, high-quality embeddings, and fast LLM responses. Keeps stack consistent.
3. Streamlit – Rapid prototyping, easy to deploy, low barrier to use for non-technical evaluators.
4. LangChain – Provides glue between retrieval and LLM, simplifying orchestration.