"""
AI ATS: JD + Multi-Resume Ranking + Voice Recruiter Agent
Groq + HuggingFace Embeddings + Qdrant + Streamlit
"""

import os
import uuid
from collections import defaultdict

import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct  # [web:60]
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from streamlit_mic_recorder import speech_to_text  # [web:105][web:107]

# Load environment variables from .env
load_dotenv()

ATS_COLLECTION = "ats_resume_collection"
VECTOR_SIZE = 384  # all-MiniLM-L6-v2 embeddings dimension


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def extract_text_from_pdf(pdf_file):
    """Extract plain text from a PDF file."""
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def chunk_text(text, chunk_size=500, chunk_overlap=50):
    """Split text into overlapping chunks for RAG."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return splitter.split_text(text)


def create_ats_collection(client: QdrantClient):
    """Create (or recreate) the ATS collection in Qdrant."""
    try:
        collections = client.get_collections().collections
        names = [c.name for c in collections]
        if ATS_COLLECTION in names:
            client.delete_collection(collection_name=ATS_COLLECTION)

        client.create_collection(
            collection_name=ATS_COLLECTION,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
    except Exception as e:
        st.error(f"Error creating ATS collection: {e}")
        raise e


def store_jd_and_resumes(client: QdrantClient, jd_text, resume_files, embeddings):
    """
    Index one JD and multiple resumes into ATS_COLLECTION.

    JD chunks: payload doc_type='jd'
    Resume chunks: payload doc_type='resume', candidate_name, source_file
    """
    create_ats_collection(client)

    points = []
    candidate_raw_text = {}

    # 1) JD chunks
    jd_chunks = chunk_text(jd_text)
    for idx, ch in enumerate(jd_chunks):
        vec = embeddings.embed_query(ch)
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={
                    "text": ch,
                    "doc_type": "jd",
                    "chunk_id": idx,
                },
            )
        )

    # 2) Resume chunks
    for uploaded in resume_files:
        filename = uploaded.name
        candidate_name = os.path.splitext(filename)[0]

        text = extract_text_from_pdf(uploaded)
        candidate_raw_text[candidate_name] = text

        chunks = chunk_text(text)
        for idx, ch in enumerate(chunks):
            vec = embeddings.embed_query(ch)
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vec,
                    payload={
                        "text": ch,
                        "doc_type": "resume",
                        "candidate_name": candidate_name,
                        "source_file": filename,
                        "chunk_id": idx,
                    },
                )
            )

    if points:
        client.upsert(collection_name=ATS_COLLECTION, points=points)

    return candidate_raw_text


def score_candidates(client: QdrantClient, jd_text, embeddings, top_k=80):
    """
    Compute a simple fit score per candidate by comparing JD embedding
    to all resume chunks stored in ATS_COLLECTION using query_points. [web:66]
    """
    query_vec = embeddings.embed_query(jd_text)

    result = client.query_points(
        collection_name=ATS_COLLECTION,
        query=query_vec,
        limit=top_k,
        with_payload=True,
    )

    scores_per_candidate = defaultdict(list)

    for scored_point in result.points:
        payload = scored_point.payload or {}
        if payload.get("doc_type") != "resume":
            continue
        cand = payload.get("candidate_name", "Unknown")
        scores_per_candidate[cand].append(scored_point.score)

    candidate_scores = {}
    for cand, vals in scores_per_candidate.items():
        avg = sum(vals) / len(vals)
        # Heuristic normalization to 0–100
        normalized = max(0.0, min(1.0, avg)) * 100
        candidate_scores[cand] = round(normalized, 1)

    return candidate_scores


def recruiter_qa(question, client: QdrantClient, embeddings, llm, jd_text, top_k=10):
    """
    Voice/text recruiter question -> RAG answer over JD + resumes in ATS_COLLECTION.
    Uses query_points to fetch the most relevant resume chunks. [web:66]
    """
    query_vec = embeddings.embed_query(question)

    result = client.query_points(
        collection_name=ATS_COLLECTION,
        query=query_vec,
        limit=top_k,
        with_payload=True,
    )

    resume_chunks = []
    for scored_point in result.points:
        payload = scored_point.payload or {}
        if payload.get("doc_type") == "resume":
            resume_chunks.append(payload.get("text", ""))

    context = "\n\n".join(resume_chunks)

    prompt = f"""You are an AI assistant helping a recruiter screen candidates.

JOB DESCRIPTION:
{jd_text}

CANDIDATE RESUME EXCERPTS (multiple candidates mixed):
{context}

RECRUITER QUESTION:
{question}

Answer clearly and concisely based ONLY on the JD and these resumes.
If something is not supported by the data, say that it is not available.
"""

    response = llm.invoke(prompt)
    return response.content, resume_chunks


# =============================================================================
# MAIN STREAMLIT APP
# =============================================================================

def main():
    st.set_page_config(
        page_title="AI ATS with Voice Recruiter",
        page_icon="🤖",
        layout="wide",
    )

    st.title("🤖 AI ATS: JD + Multi-Resume Ranking with Voice Recruiter Agent")

    st.markdown(
        """
Upload **one Job Description (JD)** and **multiple resumes**.

The app will:
- Index everything into **Qdrant** as vectors.
- Rank candidates by semantic fit for the JD.
- Let a **recruiter talk to a voice agent** to ask questions about the pool.
"""
    )

    # Check env vars
    if not os.getenv("QDRANT_URL") or not os.getenv("QDRANT_API_KEY") or not os.getenv(
        "GROQ_API_KEY"
    ):
        st.error("❌ Missing QDRANT_URL, QDRANT_API_KEY, or GROQ_API_KEY in .env")
        st.stop()

    # Initialize backends
    try:
        with st.spinner("🔄 Connecting to Qdrant & Groq..."):
            client = QdrantClient(
                url=os.getenv("QDRANT_URL"),
                api_key=os.getenv("QDRANT_API_KEY"),
            )

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

            llm = ChatGroq(
                model="llama-3.1-70b-versatile",
                temperature=0,
                groq_api_key=os.getenv("GROQ_API_KEY"),
            )

        st.success("✅ Connected successfully to Qdrant and Groq!")
    except Exception as e:
        st.error(f"Connection error: {e}")
        st.stop()

    # =========================
    # 1. Upload JD + Resumes
    # =========================

    st.header("1️⃣ Upload JD and Resumes")

    jd_col1, jd_col2 = st.columns(2)
    with jd_col1:
        jd_pdf = st.file_uploader(
            "Upload JD (PDF)", type=["pdf"], key="jd_pdf"
        )
    with jd_col2:
        jd_text_manual = st.text_area(
            "Or paste JD text directly",
            height=180,
            placeholder="Paste the job description here...",
            key="jd_text_manual",
        )

    if jd_pdf is not None:
        jd_text = extract_text_from_pdf(jd_pdf)
    else:
        jd_text = jd_text_manual

    resumes = st.file_uploader(
        "Upload multiple candidate resumes (PDF)",
        type=["pdf"],
        accept_multiple_files=True,
        key="resumes",
    )

    if st.button("🚀 Process JD + Resumes", key="process"):
        if not jd_text.strip():
            st.error("Please provide a Job Description (JD) first.")
        elif not resumes:
            st.error("Please upload at least one candidate resume.")
        else:
            with st.spinner("Indexing JD and resumes into Qdrant and scoring..."):
                try:
                    candidate_texts = store_jd_and_resumes(
                        client, jd_text, resumes, embeddings
                    )
                    candidate_scores = score_candidates(
                        client, jd_text, embeddings, top_k=80
                    )

                    st.session_state["jd_text"] = jd_text
                    st.session_state["candidate_texts"] = candidate_texts
                    st.session_state["candidate_scores"] = candidate_scores

                    st.success(
                        "✅ JD and resumes processed successfully! See ATS ranking and use the voice agent below."
                    )
                except Exception as e:
                    st.error(f"❌ ATS processing error: {e}")

    # =========================
    # 2. ATS Ranking Section
    # =========================

    st.header("2️⃣ ATS Ranking: Shortlist Candidates")

    if "candidate_scores" in st.session_state and st.session_state["candidate_scores"]:
        scores = st.session_state["candidate_scores"]
        sorted_candidates = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        st.subheader("🏆 Ranked Shortlist")
        for rank, (candidate_name, score) in enumerate(sorted_candidates, start=1):
            st.markdown(
                f"**#{rank} – {candidate_name}** · Fit Score: `{score}/100`"
            )

        st.caption(
            "Fit scores are heuristic and based on Qdrant semantic similarity between resume chunks and the JD."
        )

        st.markdown("---")
        st.subheader("🔍 Inspect Candidate Resume Text")
        cand_names = list(st.session_state["candidate_texts"].keys())
        selected_cand = st.selectbox(
            "Select a candidate to preview resume text",
            cand_names,
            key="inspect_candidate",
        )
        if selected_cand:
            raw_text = st.session_state["candidate_texts"][selected_cand]
            st.text_area(
                f"Resume text for {selected_cand} (first 2000 chars)",
                raw_text[:2000],
                height=250,
            )
    else:
        st.info("Process a JD and resumes above to see the ranking.")

    # =========================
    # 3. Voice Recruiter Agent
    # =========================

    st.header("3️⃣ 🎙️ Voice Recruiter Agent")

    if (
        "candidate_scores" not in st.session_state
        or not st.session_state["candidate_scores"]
        or "jd_text" not in st.session_state
    ):
        st.info("First upload and process a JD + resumes, then use the voice agent.")
        return

    st.markdown(
        "Ask questions about the candidate pool. Examples:\n"
        "- Who are the best candidates for backend Python?\n"
        "- Which candidates match the required cloud skills?\n"
        "- Summarize the top 3 candidates for this role.\n"
    )

    col_voice, col_text = st.columns([1, 3])

    with col_voice:
        voice_text = speech_to_text(
            language="en", use_container_width=True, just_once=True, key="stt_ats"
        )

    with col_text:
        if voice_text:
            st.write(f"🎤 Recognized voice input: `{voice_text}`")
        text_query = st.text_input(
            "Or type your question here:",
            placeholder="e.g., Who are the top 3 candidates with React and Node?",
            key="recruiter_text_q",
        )

    final_question = text_query.strip() if text_query.strip() else (voice_text or "").strip()

    if st.button("💬 Ask Recruiter Agent", key="ask_recruiter"):
        if not final_question:
            st.warning("Please speak or type a question first.")
        else:
            with st.spinner("🤔 Analyzing candidates and JD..."):
                try:
                    answer, used_chunks = recruiter_qa(
                        final_question,
                        client,
                        embeddings,
                        llm,
                        st.session_state["jd_text"],
                    )
                    st.markdown("### 💡 Answer:")
                    st.success(answer)

                    with st.expander("🔎 View Resume Excerpts Used for This Answer"):
                        for i, ch in enumerate(used_chunks, start=1):
                            st.markdown(f"**📄 Excerpt {i}:**")
                            st.info(ch)
                except Exception as e:
                    st.error(f"❌ Error in recruiter agent: {e}")


# =============================================================================
# RUN APP
# =============================================================================

if __name__ == "__main__":
    main()
