import os
import streamlit as st
import tempfile
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import re
import uuid
import glob
# Updated imports for ChromaDB
from chromadb import PersistentClient  # New import

# --- Configuration ---
st.set_page_config(layout="wide")

# PDF directory and index paths
PDF_DIR = "pdf"
INDEX_DIR = "index"
CHROMA_DB_PATH = os.path.join(INDEX_DIR, "chroma_db")

# --- Environment Variable Loading ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("🔴 GOOGLE_API_KEY not found in environment variables! Please set it in your .env file.")
    st.stop()

# --- Custom CSS ---
st.markdown("""
    <style>
    .main .block-container {
        max-width: 100% !important;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #2c3e50;
        color: white !important;
        border-left: 5px solid #3498db;
    }
    .bot-message {
        background-color: #e6f3ff;
        color: #000000 !important;
        border-left: 5px solid #2ecc71;
    }
    .user-message b, .user-message p, .user-message span {
        color: white !important;
    }
    .bot-message b, .bot-message p, .bot-message span, .bot-message div {
        color: #000000 !important;
    }
    .st-expander {
        color: #000000 !important;
    }
    .chat-history-container {
    max-height: 500px; 
    overflow-y: auto;
    padding-right: 15px; 
    margin-bottom: 1rem;
}
    </style>
""", unsafe_allow_html=True)

# --- Create necessary directories ---
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

# --- Model Initialization with Streamlit Caching ---
@st.cache_resource
def load_embedding_model():
    """Loads the HuggingFace sentence transformer embedding model."""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def load_llm_model():
    """Loads the Google Gemini LLM."""
    return ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3, convert_system_message_to_human=True)

embedding_model = load_embedding_model()
llm = load_llm_model()

if embedding_model is None or llm is None:
    st.error("🔴 Failed to load models. Please check your API keys and model configurations.")
    st.stop()

# --- Title ---
# st.markdown("""
#     <h1 style='text-align: center;'>📄 Chat with Documents using ChromaDB + Gemini</h1>
# """, unsafe_allow_html=True)

st.markdown("""
    <h1 style='text-align: center;'>📄 Onco-Med ChatBot</h1>
""", unsafe_allow_html=True)
# --- Helper functions ---
def extract_answer_only(text):
    """Extracts the answer part from the LLM's response if prefixed."""
    match = re.search(r"Answer:\s*(.*)", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()

def get_session_index_path(session_id):
    """Generates a unique path for session-specific ChromaDB instances."""
    return os.path.join(INDEX_DIR, f"session_{session_id}_chroma_db")

def load_document_from_path(file_path, file_type):
    """Loads documents from a given file path based on file type."""
    if file_type == "pdf":
        loader = PyPDFLoader(file_path)
        return loader.load()
    elif file_type in ["docx", "doc"]:
        loader = UnstructuredWordDocumentLoader(file_path)
        return loader.load()
    else:
        st.error(f"Unsupported file type: {file_type} for file {file_path}")
        return []

def get_documents_in_directory(directory_path):
    """Get list of PDF and Word documents in the specified directory."""
    pdf_files = glob.glob(os.path.join(directory_path, "*.pdf"))
    doc_files = glob.glob(os.path.join(directory_path, "*.doc")) + glob.glob(os.path.join(directory_path, "*.docx"))
    return pdf_files + doc_files

# --- New ChromaDB client helper function ---
def get_chroma_client(path):
    """Creates a persistent ChromaDB client with the new API."""
    return PersistentClient(path=path)

def process_directory_documents(directory_to_scan, _embedding_model, chunk_size=400, chunk_overlap=50):
    """
    Processes all documents in the specified directory, creates embeddings,
    and saves a ChromaDB instance.
    """
    source_documents = get_documents_in_directory(directory_to_scan)
    if not source_documents:
        return None, 0

    all_chunks = []
    files_processed_count = 0

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, file_path in enumerate(source_documents):
        status_text.text(f"Processing file: {os.path.basename(file_path)} ({i+1}/{len(source_documents)})")
        try:
            file_type = os.path.splitext(file_path)[1].lower().lstrip('.')
            loaded_docs = load_document_from_path(file_path, file_type)

            if not loaded_docs:
                st.warning(f"Could not load document: {file_path}")
                continue

            # Refined Chunking
            splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            chunks = splitter.split_documents(loaded_docs)

            for chunk in chunks:
                if not chunk.metadata:
                    chunk.metadata = {}
                chunk.metadata['source'] = os.path.basename(file_path)

            all_chunks.extend(chunks)
            files_processed_count += 1
        except Exception as e:
            st.error(f"Error processing {file_path}: {str(e)}")
        progress_bar.progress((i + 1) / len(source_documents))

    status_text.text("Creating ChromaDB index...")
    if all_chunks:
        # Use updated ChromaDB client
        client = get_chroma_client(path=CHROMA_DB_PATH)

        chroma_db = Chroma.from_documents(
            all_chunks,
            _embedding_model,
            client=client
        )
        status_text.text(f"ChromaDB index created and saved with {files_processed_count} documents.")
        return chroma_db, files_processed_count
    else:
        status_text.text("No chunks were generated from the documents.")
        return None, 0

# --- Streamlit Caching for ChromaDB index loading ---
@st.cache_resource
def load_chroma_db_from_disk(path, _embedding_model):
    """Loads ChromaDB index from disk if it exists."""
    try:
        # Updated ChromaDB client
        client = get_chroma_client(path=path)
        return Chroma(
            client=client,
            embedding_function=_embedding_model
        )
    except Exception as e:
        st.error(f"Error loading ChromaDB index from {path}: {e}. Will try to rebuild.")
        return None

# --- Function to clear conversation ---
def clear_conversation():
    """Function to clear conversation history without directly modifying input widget state"""
    st.session_state.conversation_history = []
    # We don't modify user_query_input here

# --- Initialize session state ---
if 'user_query_input' not in st.session_state:
    st.session_state.user_query_input = ""
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if 'chroma_db' not in st.session_state:
    st.session_state.chroma_db = None
if 'files_processed' not in st.session_state:
    st.session_state.files_processed = 0
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# --- Load or Process Documents on Startup ---
if st.session_state.chroma_db is None:
    with st.spinner("🔄 Loading document index..."):
        st.session_state.chroma_db = load_chroma_db_from_disk(CHROMA_DB_PATH, embedding_model)

    if st.session_state.chroma_db:
        initial_docs_in_dir = get_documents_in_directory(PDF_DIR)
        st.session_state.files_processed = len(initial_docs_in_dir)
        st.success(f"✅ Preloaded index with approximately {st.session_state.files_processed} documents loaded from disk!")
    else:
        st.info("Preloaded index not found or failed to load. Processing documents from PDF directory to create a new index. This might take some time...")
        chroma_db_created, num_files_processed = process_directory_documents(PDF_DIR, embedding_model)
        if chroma_db_created:
            st.session_state.chroma_db = chroma_db_created
            st.session_state.files_processed = num_files_processed
            st.success(f"✅ Directory documents processed and index created with {num_files_processed} documents.")
        else:
            st.warning("⚠️ No documents were found or processed in the PDF directory. Please add documents to the 'pdf' folder.")

# --- Two-column layout ---
left_col, right_col = st.columns([3, 5])

# --- Left column: File upload for additional documents ---
with left_col:
    st.subheader("📁 Document Status & Uploads")
    st.write(f"**Current Total Documents Indexed: {st.session_state.files_processed}**")
    if st.session_state.chroma_db and st.session_state.files_processed > 0:
        st.markdown(f"<span style='color:green'>✅ Index active with {st.session_state.files_processed} documents.</span>", unsafe_allow_html=True)
    elif not get_documents_in_directory(PDF_DIR):
        st.warning(f"⚠️ No documents found in the '{PDF_DIR}' directory. Please add some or upload below.")

    st.markdown("---")
    st.write("**Upload additional PDF/Word documents (max 10 files):**")
    uploaded_files = st.file_uploader(
        "Choose files to add to the current session's knowledge base",
        type=["pdf", "docx", "doc"],
        accept_multiple_files=True,
        key="file_uploader"
    )

    if uploaded_files:
        if len(uploaded_files) > 10:
            st.error("🚫 You can upload a maximum of 10 additional files at a time.")
        else:
            if st.button("➕ Process Uploaded Documents", key="process_uploaded_button"):
                with st.spinner("⏳ Processing uploaded documents..."):
                    all_new_chunks = []
                    additional_processed_count = 0
                    temp_file_paths = []

                    for uploaded_file in uploaded_files:
                        file_type = uploaded_file.name.split('.')[-1].lower()
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_type}") as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_path = tmp_file.name
                            temp_file_paths.append(tmp_path)

                        try:
                            loaded_docs = load_document_from_path(tmp_path, file_type)
                            if not loaded_docs:
                                st.warning(f"Could not load uploaded file: {uploaded_file.name}")
                                continue

                            # Refined Chunking for uploaded files as well
                            splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
                            chunks = splitter.split_documents(loaded_docs)

                            for chunk in chunks:
                                if not chunk.metadata:
                                    chunk.metadata = {}
                                chunk.metadata['source'] = f"UPLOADED: {uploaded_file.name}"

                            all_new_chunks.extend(chunks)
                            additional_processed_count += 1
                        except Exception as e:
                            st.error(f"Error processing {uploaded_file.name}: {str(e)}")

                    if all_new_chunks:
                        # Updated ChromaDB client
                        client = get_chroma_client(path=CHROMA_DB_PATH)

                        if st.session_state.chroma_db:
                            st.session_state.chroma_db.add_documents(all_new_chunks)
                            st.session_state.files_processed += additional_processed_count
                            st.success(f"✅ {additional_processed_count} uploaded documents added to the index!")
                        else:
                            st.session_state.chroma_db = Chroma.from_documents(
                                all_new_chunks,
                                embedding_model,
                                client=client,
                            )
                            st.session_state.files_processed = additional_processed_count
                            st.success(f"✅ Index created with {additional_processed_count} uploaded documents.")
                    else:
                        st.error("No content could be processed from the uploaded files.")

                    for path in temp_file_paths:
                        try:
                            os.unlink(path)
                        except Exception as e:
                            st.warning(f"Could not delete temp file {path}: {e}")
                    st.rerun()
    
    # Add Clear Conversation button to left column
    if st.button("🧹 Clear Conversation", key="clear_chat_button"):
        clear_conversation()
        st.rerun()  # Reload the app without modifying widget state directly

# --- Right column: Chat interface ---
with right_col:
    st.subheader("💬 Chat with Your Documents")
    if st.session_state.chroma_db:
        chat_container = st.container()
        with chat_container:
            st.markdown('<div class="chat-history-container">', unsafe_allow_html=True)
            for i, (query, answer) in enumerate(st.session_state.conversation_history):
                st.markdown(f'<div class="chat-message user-message"><b>You:</b><br>{query}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="chat-message bot-message"><b>Gemini:</b><br>{answer}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)  # Close the container properly

        # Use a form for question submission to better handle input state
        with st.form(key="chat_form"):
            user_query = st.text_area(
                "Ask a question about the content of your documents:",
                key="user_query_input", # Link to session state
                height=100 # Initial height, will expand (adjust 100 as needed)
            )
            
            submit_button = st.form_submit_button("Submit Question")
            
            if submit_button and user_query:
                with st.spinner("🔎 Searching and thinking..."):
                    try:
                        # Adjust Similarity Search k
                        relevant_docs = st.session_state.chroma_db.similarity_search(user_query, k=7)

                        context_parts = []
                        for doc in relevant_docs:
                            source = doc.metadata.get('source', 'Unknown source')
                            context_parts.append(f"[Source: {source}]\n{doc.page_content}")
                        context = "\n\n---\n\n".join(context_parts)

                        prompt = f"""You are a helpful AI assistant. Answer the question based ONLY on the provided context. If the answer is not found in the context, say "I could not find an answer in the provided documents." Do not make up information.

                        Context:
                        {context}

                        Question: {user_query}
                        Answer:"""

                        response_obj = llm.invoke(prompt)
                        raw_answer = response_obj.content.strip()
                        clean_answer = extract_answer_only(raw_answer)

                        st.session_state.conversation_history.append((user_query, clean_answer))
                        st.rerun()

                    except Exception as e:
                        st.error(f"An error occurred: {e}")
    else:
        st.warning("👈 Please ensure documents are loaded or uploaded to activate the chat.")
        if not get_documents_in_directory(PDF_DIR) and not uploaded_files:
            st.info(f"Add documents to the '{PDF_DIR}' folder and restart, or upload files using the panel on the left.")