import streamlit as st
from tempfile import NamedTemporaryFile
from RAGpipeline import PDFRAGSystem
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize application state
def initialize_app_state():
    """Initialize all session state variables"""
    if "rag_system" not in st.session_state:
        try:
            st.session_state.rag_system = PDFRAGSystem()
        except Exception as e:
            st.error(f"Failed to initialize system: {str(e)}")
            st.stop()
    
    # Initialize other state variables
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

# Initialize application state
initialize_app_state()

# --- Sidebar Section ---
with st.sidebar:
    st.header("📚 Search History")
    
    if st.session_state.search_history:
        for idx, (q, a) in enumerate(reversed(st.session_state.search_history), 1):
            with st.container(border=True):
                st.markdown(f"**{idx}. {q}**")
                st.markdown(f"<div style='margin-left: 15px; color: #444;'>{a}</div>", 
                          unsafe_allow_html=True)
    else:
        st.info("No previous searches", icon="ℹ️")
    
    st.markdown("---")
    st.markdown("**How to Use**")
    st.markdown("1. Upload a PDF document\n"
                "2. Process the document\n"
                "3. Ask questions naturally")

# --- Main Interface ---
st.markdown("<h1 style='text-align: center; margin-bottom: 30px;'>PDF Query Assistant</h1>", 
          unsafe_allow_html=True)

# --- Document Upload & Processing Section ---
with st.container(border=True):
    st.markdown("### Document Upload")
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf",
        label_visibility="collapsed",
        key="file_uploader"
    )
    
    # File processing controls
    col1, col2 = st.columns([1, 3])
    with col1:
        process_btn = st.button(
            " Process PDF",
            disabled=not uploaded_file,
            help="Process the uploaded PDF for querying",
            use_container_width=True
        )
    
    with col2:
        if uploaded_file:
            if st.session_state.pdf_processed:
                st.success("Document ready for queries")
            else:
                st.warning("Document needs processing")

    # Handle file processing
    if process_btn and uploaded_file:
        try:
            with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_path = temp_file.name
            
            with st.spinner(" Analyzing document content..."):
                st.session_state.rag_system.process_pdf(temp_path)
                os.unlink(temp_path)
                
                # Update state flags
                st.session_state.pdf_processed = True
                st.session_state.current_file_hash = hash(uploaded_file.getvalue())
                
            st.rerun()
            
        except Exception as e:
            st.error(f" Processing failed: {str(e)}")
            st.session_state.pdf_processed = False
            st.stop()

# --- Query Section ---
with st.container(border=True):
    st.markdown("### Query PDF")
    
    question = st.text_input(
        "Enter your question:",
        placeholder="Type your question here...",
        label_visibility="collapsed",
        key="question_input"
    )
    
    search_btn = st.button(
        "🔍 Search Answer",
        disabled=not st.session_state.pdf_processed,
        use_container_width=True
    )
    
    if search_btn:
        if not question.strip():
            st.warning("Please enter a valid question", icon="⚠️")
        else:
            try:
                with st.spinner("🔍 Searching through document..."):
                    answer = st.session_state.rag_system.query(question)
                
                # Display results
                with st.container(border=True):
                    st.markdown("#### 📝 Answer")
                    st.markdown(f"<div style='line-height: 1.6;'>{answer}</div>", 
                              unsafe_allow_html=True)
                    
                # Update search history
                st.session_state.search_history.append((question, answer))
                
            except Exception as e:
                st.error(f"Search failed: {str(e)}", icon="❌")

# --- File Change Detection ---
if uploaded_file and st.session_state.current_file_hash:
    current_hash = hash(uploaded_file.getvalue())
    if current_hash != st.session_state.current_file_hash:
        st.session_state.pdf_processed = False
        st.session_state.search_history = []
        st.rerun()