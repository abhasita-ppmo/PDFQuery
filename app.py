import streamlit as st
from tempfile import NamedTemporaryFile
from RAGpipeline import PDFRAGSystem
import os
from dotenv import load_dotenv
import time

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
    layout="centered",
    initial_sidebar_state="collapsed"
)

initialize_app_state()


# --- Sidebar Section ---
with st.sidebar:
    st.markdown("<h2 style='text-align: center;'>Search History</h2>", unsafe_allow_html=True)

    if st.session_state.search_history:
        for idx, (q, a) in enumerate(reversed(st.session_state.search_history), 1):
            with st.container():
                st.markdown(
                    f"""
                    <div style="
                        padding: 10px;
                        margin-bottom: 10px;
                        border: 1px solid #ddd;
                        border-radius: 8px;
                        transition: transform 0.3s ease, box-shadow 0.3s ease;
                        text-align: justify;
                    "
                    onmouseover="this.style.transform='scale(1.05)'; this.style.boxShadow='0 4px 8px rgba(0, 0, 0, 0.1)';"
                    onmouseout="this.style.transform='scale(1)'; this.style.boxShadow='none';">
                        <strong>{q}</strong>
                        <p style="margin-bottom: 2px; color: #444;">{a}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    else:
        st.info("No previous searches")



# --- Main Interface ---

st.markdown(
    "<h1 style='text-align: center; margin-bottom: 30px;'>PDF Query Assistant</h1>",
    unsafe_allow_html=True
)

# --- Document Upload Section ---
with st.container():


    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf",
        label_visibility="collapsed",
        key="file_uploader"
    )


def stream_response(answer):
    placeholder = st.empty()
    displayed_text = ""

    for word in answer.split():
        displayed_text += word + " "
        placeholder.markdown(f"<div style='line-height: 1.6; font-size: 16px;'>{displayed_text}</div>", unsafe_allow_html=True)
        time.sleep(0.05)  # For speed

with st.container():
    st.session_state.search_mode = st.toggle(
        "Enable RAG (Generative Answers)",
        value=True,
        help="Toggle ON for generated answers, and OFF for direct text matches"
    )

    col1, col2 = st.columns([4, 1])
    with col1:
        question = st.text_input(
            "Enter your question:",
            placeholder="Type your question here",
            label_visibility="collapsed",
            key="question_input"
        )
    with col2:
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

                if not st.session_state.pdf_processed or current_hash != st.session_state.current_file_hash:
                    with st.spinner("Processing PDF content"):
                        with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                            temp_file.write(uploaded_file.getvalue())
                            temp_path = temp_file.name
                        try:
                            st.session_state.rag_system.process_pdf(temp_path)
                            st.session_state.pdf_processed = True
                            st.session_state.current_file_hash = current_hash
                        finally:
                            os.unlink(temp_path)

                with st.spinner("Analyzing document content"):
                    if st.session_state.search_mode:
                        answer = st.session_state.rag_system.query(question)
                    else:
                        with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                            temp_file.write(uploaded_file.getvalue())
                            temp_path = temp_file.name
                        try:
                            answer = st.session_state.rag_system.search(temp_path, question)
                        finally:
                            os.unlink(temp_path)

                if answer.startswith("Query failed:") and "No such file or directory" in answer:
                    with st.spinner("Reprocessing PDF"):
                        with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                            temp_file.write(uploaded_file.getvalue())
                            temp_path = temp_file.name
                        try:
                            st.session_state.rag_system.process_pdf(temp_path)
                            st.session_state.pdf_processed = True
                            st.session_state.current_file_hash = current_hash
                        finally:
                            os.unlink(temp_path)
                    with st.spinner("Retrying query"):
                        answer = st.session_state.rag_system.query(question)

                #st.markdown("#### Answer")
                stream_response(answer)

                st.session_state.search_history.append((question, answer))

            except Exception as e:
                st.error(f"Error processing request: {str(e)}")
                st.session_state.pdf_processed = False
