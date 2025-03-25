import streamlit as st
from tempfile import NamedTemporaryFile
from RAGpipeline import PDFRAGSystem
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def initialize_app_state():
    """Initialize all session state variables"""
    if "rag_system" not in st.session_state:
        try:
            st.session_state.rag_system = PDFRAGSystem()
        except Exception as e:
            st.error(f"Failed to initialize system: {str(e)}")
            st.stop()
    
    state_defaults = {
        "search_history": [],
        "pdf_processed": False,
        "current_file_hash": None
    }
    
    for key, value in state_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Configure page settings
st.set_page_config(
    page_title="PDF Query Assistant",
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="expanded"
)

initialize_app_state()

# --- Sidebar Section ---
with st.sidebar:
    st.header("📚 Search History")
    
    if st.session_state.search_history:
        for idx, (q, a) in enumerate(reversed(st.session_state.search_history), 1):
            with st.container():
                st.markdown(f"**{idx}. {q}**")
                st.markdown(
                    f"<div style='margin-left: 15px; color: #444;'>{a}</div>", 
                    unsafe_allow_html=True
                )
    else:
        st.info("No previous searches", icon="ℹ️")
    
    st.markdown("---")
    st.markdown("**How to Use**")
    st.markdown("1. Upload a PDF document\n2. Ask questions naturally")

# --- Main Interface ---
st.markdown(
    "<h1 style='text-align: center; margin-bottom: 30px;'>PDF Query Assistant</h1>", 
    unsafe_allow_html=True
)

# --- Document Upload Section ---
with st.container():
    st.markdown("### Document Upload")
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf",
        label_visibility="collapsed",
        key="file_uploader"
    )
    
    if uploaded_file:
        # Use file hash to determine if the uploaded file has already been processed
        if st.session_state.pdf_processed and \
           st.session_state.current_file_hash == hash(uploaded_file.getvalue()):
            st.success("Document ready for queries")
        else:
            st.info("PDF uploaded - will process during first query")

# --- Query Section ---
with st.container():
    st.markdown("### Query PDF")
    st.session_state.search_mode = st.radio(
        "Search Mode",
        ["RAG (Generative Answers)", "Generic LLM"],
        index=0,
        help="Choose between generated answers or direct text matches"
    )
    
    question = st.text_input(
        "Enter your question:",
        placeholder="Type your question here...",
        label_visibility="collapsed",
        key="question_input"
    )
    
    search_btn = st.button(
        "Search Answer",
        disabled=not uploaded_file,
        use_container_width=True
    )
    
    if search_btn:
        if not question.strip():
            st.warning("Please enter a valid question")
        else:
            try:
                current_hash = hash(uploaded_file.getvalue())

                # Process the PDF if it hasn't been processed yet or if a new file is uploaded
                if not st.session_state.pdf_processed or current_hash != st.session_state.current_file_hash:
                    with st.spinner("📄 Processing PDF content..."):
                        with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                            temp_file.write(uploaded_file.getvalue())
                            temp_path = temp_file.name
                        try:
                            st.session_state.rag_system.process_pdf(temp_path)
                            st.session_state.pdf_processed = True
                            st.session_state.current_file_hash = current_hash
                        finally:
                            os.unlink(temp_path)

                # Execute the query
                with st.spinner("🔍 Analyzing document content..."):
                    if "RAG" in st.session_state.search_mode:
                        answer = st.session_state.rag_system.query(question)
                    else:
                        # For non-RAG queries, create a temporary file and call search with pdf_path & question
                            with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                                temp_file.write(uploaded_file.getvalue())
                                temp_path = temp_file.name
                            try:
                                answer = st.session_state.rag_system.search(temp_path, question)
                            finally:
                                os.unlink(temp_path)
                
                # If error indicates a missing index file, reprocess PDF and retry the query
                if answer.startswith("Query failed:") and "No such file or directory" in answer:
                    with st.spinner("Reprocessing PDF..."):
                        with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                            temp_file.write(uploaded_file.getvalue())
                            temp_path = temp_file.name
                        try:
                            st.session_state.rag_system.process_pdf(temp_path)
                            st.session_state.pdf_processed = True
                            st.session_state.current_file_hash = current_hash
                        finally:
                            os.unlink(temp_path)
                    with st.spinner("🔍 Retrying query..."):
                        answer = st.session_state.rag_system.query(question)
                        print("Generated answer:", answer)
                
                # Display the answer
                with st.container():
                    st.markdown("#### Answer")
                    st.markdown(
                        f"<div style='line-height: 1.4;'>{answer}</div>", 
                        unsafe_allow_html=True
                    )
                
                # Update search history
                st.session_state.search_history.append((question, answer))
                
            except Exception as e:
                st.error(f"Error processing request: {str(e)}")
                st.session_state.pdf_processed = False
