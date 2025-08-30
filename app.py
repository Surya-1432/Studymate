# app.py
import os
import streamlit as st
from pdf_processor import extract_pages
from chunker import chunk_text_per_page
from embed_index import EmbedIndex
from llm_client import build_prompt, call_hf_granite
from utils import save_session_transcript, timestamp_now
import random

# ---------------- Page Configuration ----------------
st.set_page_config(page_title="StudyMate (HF LLM)", layout="wide")
st.title("StudyMate — PDF Q&A (Hugging Face LLM)")

# ---------------- Animated Background + Bubbles ----------------
num_bubbles = 30
bubbles_html = ""
for i in range(num_bubbles):
    size = random.randint(20, 60)
    left = random.randint(0, 100)
    delay = random.uniform(0, 20)
    duration = random.uniform(10, 25)
    bubbles_html += f"""
    <div class='bubble' style='width:{size}px;height:{size}px;left:{left}%;animation-delay:{delay}s;animation-duration:{duration}s'></div>
    """

st.markdown(
    f"""
    <style>
    body {{
        background: linear-gradient(135deg, #ff9a9e 0%, #fad0c4 50%, #a1c4fd 100%);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
    }}
    @keyframes gradientBG {{
        0% {{background-position:0% 50%;}}
        50% {{background-position:100% 50%;}}
        100% {{background-position:0% 50%;}}
    }}
    .bubble {{
        position: fixed;
        bottom: -100px;
        border-radius: 50%;
        background: rgba(255,255,255,0.3);
        animation: float linear infinite;
    }}
    @keyframes float {{
        0% {{transform: translateY(0px);}}
        100% {{transform: translateY(-1200px);}}
    }}
    </style>
    {bubbles_html}
    """, unsafe_allow_html=True
)

# ---------------- Paths ----------------
INDEX_PATH = "data/index/index.faiss"
META_PATH = "data/index/meta.pkl"

# ---------------- Initialize Session State ----------------
if "session_history" not in st.session_state:
    st.session_state.session_history = []

if "index_loaded" not in st.session_state:
    st.session_state.index_loaded = False

if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = True

if "index" not in st.session_state:
    st.session_state.index = EmbedIndex(index_path=INDEX_PATH, meta_path=META_PATH)
index = st.session_state.index

# ---------------- Auto-load index if exists ----------------
if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH) and not st.session_state.index_loaded:
    try:
        index.load()
        st.session_state.index_loaded = True
        st.success("Existing index loaded automatically!")
    except Exception as e:
        st.warning(f"Could not auto-load index: {e}")

# ---------------- Sidebar: Upload PDFs and Build Index ----------------
st.sidebar.header("Upload PDFs & Build Index")
uploaded_files = st.sidebar.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)
chunk_size = st.sidebar.number_input("Chunk size (words)", value=500, min_value=100)
overlap = st.sidebar.number_input("Overlap (words)", value=100, min_value=0)

if st.sidebar.button("Process & Build Index"):
    if not uploaded_files:
        st.sidebar.warning("Upload 1+ PDF to build index.")
    else:
        all_chunks = []
        for pdf in uploaded_files:
            pages = extract_pages(pdf)
            chunks = chunk_text_per_page(pdf.name, pages, chunk_size=chunk_size, overlap=overlap)
            all_chunks.extend(chunks)
        with st.spinner("Building FAISS index..."):
            index.build(all_chunks)
        st.sidebar.success("Index built & saved to disk!")
        st.session_state.index_loaded = True

# ---------------- Query Section ----------------
st.header("Ask questions about the uploaded PDFs")
question = st.text_input("Enter your question here")
k = st.slider("Number of chunks to retrieve (k)", 1, 10, 3)

if st.button("💡 Get Answer"):
    if not st.session_state.index_loaded:
        st.warning("Please build or load an index first (sidebar).")
    elif not question.strip():
        st.warning("Type a question first.")
    elif not st.session_state.model_loaded:
        st.error("LLM model failed to load. Check HF_MODEL_ID or HF_TOKEN in .env.")
    else:
        try:
            with st.spinner("Retrieving top chunks..."):
                results = index.query(question, top_k=k)
        except Exception as e:
            st.error(f"Failed to query index: {e}")
            results = []

        if results:
            try:
                prompt = build_prompt(question, results)
                with st.spinner("Calling Hugging Face LLM..."):
                    answer = call_hf_granite(prompt)
            except Exception as e:
                st.error(f"LLM call failed: {e}")
                answer = f"Error: {e}"
                st.session_state.model_loaded = False

            # Save session
            entry = {"timestamp": timestamp_now(), "question": question, "answer": answer}
            st.session_state.session_history.append(entry)

            # ---------------- Display Answer and Chunks in Columns ----------------
            col1, col2 = st.columns([3, 2])
            with col1:
                st.subheader("Answer")
                st.markdown(f"<div style='color: #ffffff; font-size:16px;'>{answer}</div>", unsafe_allow_html=True)

            with col2:
                st.subheader("Referenced chunks")
                for r in results:
                    md = r.get("metadata", {})
                    source = md.get("source", "Unknown source")
                    chunk_id = md.get("chunk_id", "N/A")
                    text_preview = md.get("text", "")
                    with st.expander(f"{source} (chunk {chunk_id}) — score {r.get('score', 0):.4f}"):
                        st.markdown(f"<span style='color:#1A8917'>{text_preview}</span>", unsafe_allow_html=True)
        else:
            st.warning("No chunks found to answer your question.")

# ---------------- Session History ----------------
st.header("Session history")
for entry in reversed(st.session_state.session_history):
    st.markdown(f"**{entry['timestamp']}**  \n**Q:** {entry['question']}  \n**A:** {entry['answer']}")

# ---------------- Download Transcript ----------------
if st.button("📥 Download transcript"):
    filename = save_session_transcript(st.session_state.session_history)
    with open(filename, "rb") as f:
        st.download_button("Download Q&A transcript", f, file_name=os.path.basename(filename))
