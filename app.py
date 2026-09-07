from pathlib import Path
import textwrap, zipfile, os, json

root = Path("/mnt/data/llm_study_assistant")
(root / "utils").mkdir(parents=True, exist_ok=True)
(root / "data" / "uploads").mkdir(parents=True, exist_ok=True)
(root / "faiss_index").mkdir(parents=True, exist_ok=True)

files = {}

files["app.py"] = r'''
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from utils.config import (
    APP_TITLE,
    DEFAULT_OLLAMA_MODEL,
    EMBEDDING_MODEL,
    INDEX_DIR,
    UPLOAD_DIR,
)
from utils.pdf_utils import save_uploaded_file, extract_pdf_documents
from utils.rag import (
    build_vector_store,
    load_vector_store,
    answer_with_rag,
    get_ollama_models,
)
from utils.web_search import web_search, format_search_results
from utils.s3_utils import s3_enabled, upload_file_to_s3, list_s3_pdfs

load_dotenv()

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📚",
    layout="wide",
)

Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
Path(INDEX_DIR).mkdir(parents=True, exist_ok=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store_ready" not in st.session_state:
    st.session_state.vector_store_ready = False


def render_sidebar():
    st.sidebar.title("⚙️ Settings")

    try:
        models = get_ollama_models()
    except Exception:
        models = []

    if models:
        default_index = (
            models.index(DEFAULT_OLLAMA_MODEL)
            if DEFAULT_OLLAMA_MODEL in models
            else 0
        )
        model = st.sidebar.selectbox(
            "Ollama model",
            options=models,
            index=default_index,
        )
    else:
        model = st.sidebar.text_input(
            "Ollama model",
            value=DEFAULT_OLLAMA_MODEL,
            help="Example: llama3.2:3b, gemma3:4b, qwen2.5:3b",
        )

    top_k = st.sidebar.slider("Retrieved chunks", 2, 10, 4)
    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.1,
    )
    use_web_fallback = st.sidebar.checkbox(
        "Use web search when PDF context is weak",
        value=True,
    )

    st.sidebar.caption(f"Embeddings: {EMBEDDING_MODEL}")
    st.sidebar.caption("Embeddings run on CPU by default.")

    return model, top_k, temperature, use_web_fallback


def process_pdfs(uploaded_files):
    all_documents = []
    saved_paths = []

    progress = st.progress(0, text="Reading PDFs...")

    for i, uploaded_file in enumerate(uploaded_files):
        local_path = save_uploaded_file(uploaded_file, UPLOAD_DIR)
        saved_paths.append(local_path)

        docs = extract_pdf_documents(local_path)
        all_documents.extend(docs)

        progress.progress(
            (i + 1) / len(uploaded_files),
            text=f"Processed {uploaded_file.name}",
        )

        if s3_enabled():
            try:
                upload_file_to_s3(local_path)
            except Exception as exc:
                st.warning(f"S3 upload skipped for {uploaded_file.name}: {exc}")

    progress.empty()

    if not all_documents:
        raise ValueError("No readable text was found in the uploaded PDFs.")

    with st.spinner("Creating embeddings and FAISS index..."):
        build_vector_store(all_documents)

    st.session_state.vector_store_ready = True
    return len(all_documents), saved_paths


st.title("📚 LLM Study Assistant")
st.write(
    "Upload study PDFs, build a local RAG knowledge base, and ask questions "
    "using an Ollama model."
)

model_name, top_k, temperature, use_web_fallback = render_sidebar()

tab_chat, tab_upload, tab_cloud = st.tabs(
    ["💬 Study Chat", "📄 PDF Knowledge Base", "☁️ S3 Storage"]
)

with tab_upload:
    st.subheader("Build / update your study knowledge base")

    uploaded_files = st.file_uploader(
        "Upload one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if st.button(
        "Process PDFs",
        type="primary",
        disabled=not uploaded_files,
    ):
        try:
            chunk_count, saved_paths = process_pdfs(uploaded_files)
            st.success(
                f"Knowledge base created successfully from {len(saved_paths)} "
                f"PDF(s), producing {chunk_count} page documents before chunking."
            )
        except Exception as exc:
            st.error(f"Could not process PDFs: {exc}")

    if Path(INDEX_DIR).exists() and any(Path(INDEX_DIR).iterdir()):
        st.info("A saved FAISS index is available on this computer.")

with tab_chat:
    st.subheader("Ask your study material")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask a question about your PDFs...")

    if question:
        st.session_state.messages.append(
            {"role": "user", "content": question}
        )
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                vector_store = load_vector_store()

                if vector_store is not None:
                    response = answer_with_rag(
                        question=question,
                        vector_store=vector_store,
                        model_name=model_name,
                        top_k=top_k,
                        temperature=temperature,
                    )

                    answer = response["answer"]
                    sources = response["sources"]
                    weak_context = response["weak_context"]

                    if weak_context and use_web_fallback:
                        search_results = web_search(question, max_results=5)
                        if search_results:
                            web_context = format_search_results(search_results)
                            response = answer_with_rag(
                                question=question,
                                vector_store=vector_store,
                                model_name=model_name,
                                top_k=top_k,
                                temperature=temperature,
                                extra_context=web_context,
                            )
                            answer = response["answer"]
                            answer += "\n\n**Web fallback was used for extra context.**"
                elif use_web_fallback:
                    search_results = web_search(question, max_results=5)
                    if search_results:
                        web_context = format_search_results(search_results)
                        from utils.rag import answer_without_vector_store
                        answer = answer_without_vector_store(
                            question=question,
                            context=web_context,
                            model_name=model_name,
                            temperature=temperature,
                        )
                        sources = []
                    else:
                        answer = (
                            "No FAISS knowledge base exists yet and web search "
                            "did not return results. Upload PDFs first."
                        )
                        sources = []
                else:
                    answer = "Upload and process PDFs first."
                    sources = []

                st.markdown(answer)

                if sources:
                    with st.expander("Sources from your PDFs"):
                        for source in sources:
                            page = source.get("page")
                            page_text = (
                                f" — page {page + 1}"
                                if isinstance(page, int)
                                else ""
                            )
                            st.write(f"• {source.get('source', 'PDF')}{page_text}")

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )

            except Exception as exc:
                msg = (
                    f"Error: {exc}\n\n"
                    "Check that Ollama is running and that the selected model "
                    "is installed."
                )
                st.error(msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": msg}
                )

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

with tab_cloud:
    st.subheader("Optional AWS S3 PDF backup")

    if not s3_enabled():
        st.info(
            "S3 is disabled. Add AWS credentials and AWS_S3_BUCKET to .env "
            "if you want cloud PDF backup."
        )
    else:
        st.success("S3 configuration detected.")
        if st.button("List PDFs in S3"):
            try:
                objects = list_s3_pdfs()
                if not objects:
                    st.write("No PDFs found.")
                else:
                    for obj in objects:
                        st.write(
                            f"• {obj['Key']} "
                            f"({obj['Size'] / 1024:.1f} KB)"
                        )
            except Exception as exc:
                st.error(f"S3 error: {exc}")
'''

files["utils/__init__.py"] = r'''
"""Utility package for the LLM Study Assistant."""
'''

files["utils/config.py"] = r'''
import os

APP_TITLE = "LLM Study Assistant"

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "data/uploads")
INDEX_DIR = os.getenv("INDEX_DIR", "faiss_index")

DEFAULT_OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "llama3.2:3b",
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# Similarity score threshold for deciding whether retrieved context is weak.
# FAISS similarity_search_with_score returns a distance; smaller is better.
WEAK_CONTEXT_DISTANCE = float(
    os.getenv("WEAK_CONTEXT_DISTANCE", "1.25")
)
'''

files["utils/pdf_utils.py"] = r'''
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from pypdf import PdfReader


def save_uploaded_file(uploaded_file, upload_dir: str) -> str:
    upload_path = Path(upload_dir)
    upload_path.mkdir(parents=True, exist_ok=True)

    safe_name = Path(uploaded_file.name).name
    destination = upload_path / safe_name

    with destination.open("wb") as file:
        file.write(uploaded_file.getbuffer())

    return str(destination)


def extract_pdf_documents(pdf_path: str) -> List[Document]:
    reader = PdfReader(pdf_path)
    documents = []

    for page_number, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = text.strip()

        if not text:
            continue

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": Path(pdf_path).name,
                    "path": str(pdf_path),
                    "page": page_number,
                },
            )
        )

    return documents
'''

files["utils/rag.py"] = r'''
import json
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    INDEX_DIR,
    WEAK_CONTEXT_DISTANCE,
)


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def split_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def build_vector_store(documents: List[Document]) -> FAISS:
    chunks = split_documents(documents)

    if not chunks:
        raise ValueError("No text chunks were created from the PDFs.")

    embeddings = get_embeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)

    index_path = Path(INDEX_DIR)
    index_path.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(index_path))

    return vector_store


def load_vector_store() -> Optional[FAISS]:
    index_path = Path(INDEX_DIR)

    index_file = index_path / "index.faiss"
    metadata_file = index_path / "index.pkl"

    if not index_file.exists() or not metadata_file.exists():
        return None

    embeddings = get_embeddings()

    return FAISS.load_local(
        str(index_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def get_llm(model_name: str, temperature: float = 0.2) -> OllamaLLM:
    return OllamaLLM(
        model=model_name,
        temperature=temperature,
    )


def get_ollama_models() -> List[str]:
    """
    Reads Ollama's local HTTP API without adding another dependency.
    Ollama normally exposes this endpoint at localhost:11434.
    """
    request = urllib.request.Request(
        "http://localhost:11434/api/tags",
        headers={"Accept": "application/json"},
    )

    with urllib.request.urlopen(request, timeout=2) as response:
        data = json.loads(response.read().decode("utf-8"))

    return [
        model["name"]
        for model in data.get("models", [])
        if model.get("name")
    ]


def _build_context(scored_docs):
    sections = []
    sources = []

    for i, (doc, score) in enumerate(scored_docs, start=1):
        source = doc.metadata.get("source", "PDF")
        page = doc.metadata.get("page")

        citation_label = source
        if isinstance(page, int):
            citation_label += f", page {page + 1}"

        sections.append(
            f"[PDF SOURCE {i}: {citation_label}]\n{doc.page_content}"
        )

        sources.append(
            {
                "source": source,
                "page": page,
                "distance": float(score),
            }
        )

    return "\n\n".join(sections), sources


def _prompt(question: str, context: str) -> str:
    return f"""
You are a careful study assistant.

Rules:
1. Answer primarily from the supplied context.
2. If the context does not contain enough information, clearly say what is missing.
3. Do not invent facts, page numbers, formulas, citations, or quotations.
4. Explain difficult ideas simply and step-by-step.
5. Use short headings and bullet points when useful.
6. When PDF source labels are supplied, refer to them naturally in the answer.
7. If web context is supplied, distinguish it from the student's uploaded PDFs.

CONTEXT
-------
{context}

QUESTION
--------
{question}

ANSWER
------
""".strip()


def answer_with_rag(
    question: str,
    vector_store: FAISS,
    model_name: str,
    top_k: int = 4,
    temperature: float = 0.2,
    extra_context: str = "",
) -> Dict:
    scored_docs = vector_store.similarity_search_with_score(
        question,
        k=top_k,
    )

    pdf_context, sources = _build_context(scored_docs)

    combined_context = pdf_context
    if extra_context:
        combined_context += (
            "\n\n[WEB SEARCH CONTEXT]\n" + extra_context
        )

    llm = get_llm(model_name, temperature)
    answer = llm.invoke(_prompt(question, combined_context))

    distances = [float(score) for _, score in scored_docs]
    best_distance = min(distances) if distances else float("inf")

    return {
        "answer": answer,
        "sources": sources,
        "weak_context": best_distance > WEAK_CONTEXT_DISTANCE,
        "best_distance": best_distance,
    }


def answer_without_vector_store(
    question: str,
    context: str,
    model_name: str,
    temperature: float = 0.2,
) -> str:
    llm = get_llm(model_name, temperature)
    return llm.invoke(_prompt(question, context))
'''

files["utils/web_search.py"] = r'''
from typing import Dict, List

from ddgs import DDGS


def web_search(query: str, max_results: int = 5) -> List[Dict]:
    """
    Search the web using the DDGS package.
    Returns normalized dictionaries so the rest of the app does not depend
    on a particular backend's field names.
    """
    try:
        results = DDGS().text(
            query,
            max_results=max_results,
        )
    except Exception:
        return []

    normalized = []

    for item in results or []:
        normalized.append(
            {
                "title": item.get("title", ""),
                "url": item.get("href")
                or item.get("url", ""),
                "body": item.get("body")
                or item.get("snippet", ""),
            }
        )

    return normalized


def format_search_results(results: List[Dict]) -> str:
    parts = []

    for i, result in enumerate(results, start=1):
        parts.append(
            f"[WEB RESULT {i}]\n"
            f"Title: {result.get('title', '')}\n"
            f"URL: {result.get('url', '')}\n"
            f"Summary: {result.get('body', '')}"
        )

    return "\n\n".join(parts)
'''

files["utils/s3_utils.py"] = r'''
import os
from pathlib import Path
from typing import Dict, List

import boto3
from botocore.exceptions import BotoCoreError, ClientError


def s3_enabled() -> bool:
    return bool(os.getenv("AWS_S3_BUCKET"))


def _client():
    region = os.getenv("AWS_REGION") or None
    return boto3.client("s3", region_name=region)


def upload_file_to_s3(local_path: str) -> str:
    bucket = os.getenv("AWS_S3_BUCKET")
    if not bucket:
        raise RuntimeError("AWS_S3_BUCKET is not configured.")

    prefix = os.getenv("AWS_S3_PREFIX", "study-pdfs").strip("/")
    file_name = Path(local_path).name
    key = f"{prefix}/{file_name}" if prefix else file_name

    try:
        _client().upload_file(local_path, bucket, key)
    except (BotoCoreError, ClientError) as exc:
        raise RuntimeError(str(exc)) from exc

    return key


def list_s3_pdfs() -> List[Dict]:
    bucket = os.getenv("AWS_S3_BUCKET")
    if not bucket:
        raise RuntimeError("AWS_S3_BUCKET is not configured.")

    prefix = os.getenv("AWS_S3_PREFIX", "study-pdfs").strip("/")
    prefix_query = f"{prefix}/" if prefix else ""

    objects = []
    continuation_token = None

    while True:
        kwargs = {
            "Bucket": bucket,
            "Prefix": prefix_query,
        }
        if continuation_token:
            kwargs["ContinuationToken"] = continuation_token

        response = _client().list_objects_v2(**kwargs)

        for obj in response.get("Contents", []):
            if obj["Key"].lower().endswith(".pdf"):
                objects.append(obj)

        if not response.get("IsTruncated"):
            break

        continuation_token = response.get("NextContinuationToken")

    return objects
'''

files["requirements.txt"] = r'''
streamlit>=1.40
python-dotenv>=1.0
pypdf>=5.0
faiss-cpu>=1.9
sentence-transformers>=3.0
langchain-core>=0.3
langchain-community>=0.3
langchain-text-splitters>=0.3
langchain-huggingface>=0.1
langchain-ollama>=0.2
ddgs>=9.0
boto3>=1.35
'''

files[".env.example"] = r'''
# ---------------------------
# Local LLM
# ---------------------------
OLLAMA_MODEL=llama3.2:3b

# A small CPU-friendly embedding model
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# ---------------------------
# RAG
# ---------------------------
UPLOAD_DIR=data/uploads
INDEX_DIR=faiss_index
CHUNK_SIZE=1000
CHUNK_OVERLAP=150

# FAISS distance: smaller means more similar.
# Tune this value for your documents/questions.
WEAK_CONTEXT_DISTANCE=1.25

# ---------------------------
# Optional AWS S3
# ---------------------------
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_REGION=ap-south-1
AWS_S3_BUCKET=
AWS_S3_PREFIX=study-pdfs
'''

files[".gitignore"] = r'''
.venv/
venv/
__pycache__/
*.pyc
.env

data/uploads/*
!data/uploads/.gitkeep

faiss_index/*
!faiss_index/.gitkeep

.DS_Store
'''

files["README.md"] = r'''
# LLM Study Assistant

A local study assistant built with:

- Streamlit
- Ollama
- LangChain
- Hugging Face sentence-transformer embeddings
- FAISS
- PDF extraction with pypdf
- DDGS web-search fallback
- Optional AWS S3 PDF backup

## Features

1. Upload one or more PDFs.
2. Extract text page-by-page.
3. Split the content into overlapping chunks.
4. Generate local CPU embeddings.
5. Store vectors in FAISS.
6. Retrieve relevant chunks for each question.
7. Ask a local Ollama LLM to answer from the retrieved context.
8. Optionally use web-search context when PDF retrieval appears weak.
9. Optionally back up uploaded PDFs to AWS S3.

## Recommended setup for Windows / Intel integrated graphics

Python 3.11 is a safe choice for this project.

### 1. Install Ollama

Install Ollama for Windows and start it.

Then open PowerShell:

```powershell
ollama --version
ollama pull llama3.2:3b
ollama listx