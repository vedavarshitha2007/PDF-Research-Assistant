
import os
import tempfile
import streamlit as st

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessing import split_documents
from src.llm import generate_answer_from_documents

load_dotenv()

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📚",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS - DARK BLUE THEME
# --------------------------------------------------

st.markdown("""
<style>

/* Main background */
.stApp,
[data-testid="stMain"] {
    background-color: #0B1120 !important;
    color: #F1F5F9 !important;
}

/* REMOVE TOP WHITE HEADER AND TOOLBAR */
header[data-testid="stHeader"] {
    display: none !important;
}

[data-testid="stToolbar"] {
    display: none !important;
}

[data-testid="stDecoration"] {
    display: none !important;
}

/* Remove unnecessary top spacing */
.block-container {
    padding-top: 1rem !important;
}

/* Headings */
h1, h2, h3 {
    color: #60A5FA !important;
    font-weight: 700 !important;
}

/* General text */
p, label, li {
    color: #E2E8F0 !important;
}

/* Header */
.main-title {
    font-size: 38px;
    font-weight: 800;
    color: #60A5FA;
    text-align: center;
    letter-spacing: 1px;
    padding-top: 15px;
}

.subtitle {
    text-align: center;
    color: #94A3B8;
    font-size: 15px;
    margin-bottom: 30px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #111C30 !important;
    border-right: 1px solid #263750;
}

[data-testid="stSidebar"] * {
    color: #E2E8F0 !important;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #60A5FA !important;
}

/* TEXT INPUT AND QUESTION TEXTAREA */
.stTextInput input,
.stTextArea textarea,
[data-testid="stTextArea"] textarea {
    background-color: #172338 !important;
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    border: 2px solid #3B82F6 !important;
    border-radius: 10px !important;
    padding: 12px !important;
    caret-color: #60A5FA !important;
    font-size: 15px !important;
    outline: none !important;
    box-shadow: none !important;
}

/* Keep question box dark in every state */
.stTextArea textarea:hover,
.stTextArea textarea:focus,
.stTextArea textarea:active,
[data-testid="stTextArea"] textarea:hover,
[data-testid="stTextArea"] textarea:focus,
[data-testid="stTextArea"] textarea:active {
    background-color: #172338 !important;
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    caret-color: #60A5FA !important;
    border: 2px solid #3B82F6 !important;
    outline: none !important;
    box-shadow: none !important;
}

/* Question box height */
.stTextArea textarea {
    min-height: 120px !important;
}

/* Input labels */
.stTextInput label,
.stTextArea label {
    color: #93C5FD !important;
    font-weight: 700 !important;
}

/* Placeholder text */
.stTextInput input::placeholder,
.stTextArea textarea::placeholder,
[data-testid="stTextArea"] textarea::placeholder {
    color: #A7B7CE !important;
    -webkit-text-fill-color: #A7B7CE !important;
    opacity: 1 !important;
}

/* Selected text */
.stTextArea textarea::selection {
    background-color: #2563EB !important;
    color: #FFFFFF !important;
}

/* Main buttons */
.stButton button {
    background-color: #2563EB !important;
    color: #FFFFFF !important;
    border: 1px solid #60A5FA !important;
    border-radius: 9px !important;
    font-weight: 700 !important;
    padding: 10px 18px !important;
    transition: 0.2s ease;
}

.stButton button p,
.stButton button span {
    color: #FFFFFF !important;
}

.stButton button:hover {
    background-color: #1D4ED8 !important;
    border-color: #93C5FD !important;
}

/* PDF UPLOADER */
[data-testid="stFileUploader"] {
    background-color: #172338 !important;
    border: 1px solid #3B82F6 !important;
    border-radius: 12px !important;
    padding: 15px !important;
}

[data-testid="stFileUploader"] section {
    background-color: #1E293B !important;
    border: 1px dashed #60A5FA !important;
    border-radius: 9px !important;
}

[data-testid="stFileUploader"] button {
    background-color: #2563EB !important;
    color: #FFFFFF !important;
    border: 1px solid #60A5FA !important;
    border-radius: 7px !important;
    font-weight: 700 !important;
    opacity: 1 !important;
}

[data-testid="stFileUploader"] button *,
[data-testid="stFileUploader"] button p,
[data-testid="stFileUploader"] button span {
    color: #FFFFFF !important;
    fill: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}

[data-testid="stFileUploader"] button:hover {
    background-color: #1D4ED8 !important;
}

[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] section span {
    color: #CBD5E1 !important;
}

[data-testid="stFileUploader"] [data-testid="stFileUploaderFileName"] {
    color: #FFFFFF !important;
}

/* Expanders */
[data-testid="stExpander"] {
    background-color: #172338 !important;
    border: 1px solid #334155 !important;
    border-radius: 10px !important;
}

[data-testid="stExpander"] summary {
    color: #93C5FD !important;
    font-weight: 700 !important;
}

[data-testid="stExpander"] p,
[data-testid="stExpander"] li {
    color: #E2E8F0 !important;
}

/* Alerts */
[data-testid="stAlert"] {
    background-color: #172338 !important;
    border-radius: 9px !important;
}

/* Metrics */
[data-testid="stMetric"] {
    background-color: #172338 !important;
    border: 1px solid #334155 !important;
    padding: 15px !important;
    border-radius: 10px !important;
}

[data-testid="stMetricLabel"],
[data-testid="stMetricValue"] {
    color: #FFFFFF !important;
}

/* Dividers */
hr {
    border-color: #334155 !important;
}

/* Footer */
.footer {
    text-align: center;
    color: #94A3B8;
    font-size: 13px;
    padding: 15px;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "chunks" not in st.session_state:
    st.session_state.chunks = None

if "vectorizer" not in st.session_state:
    st.session_state.vectorizer = None

if "chunk_vectors" not in st.session_state:
    st.session_state.chunk_vectors = None

if "document_name" not in st.session_state:
    st.session_state.document_name = None

if "history" not in st.session_state:
    st.session_state.history = []


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<div class="main-title">📚 PDF RAG ASSISTANT</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Upload documents • Ask questions • Get AI-powered answers'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("📁 Document Information")
st.sidebar.divider()

if st.session_state.document_name:

    st.sidebar.success("Document Loaded")

    st.sidebar.markdown(
        f"**File Name:**  \n{st.session_state.document_name}"
    )

    if st.session_state.chunks:
        st.sidebar.metric(
            "Total Chunks",
            len(st.session_state.chunks)
        )

else:
    st.sidebar.info("Upload a PDF to get started.")

st.sidebar.divider()

st.sidebar.markdown("### ⚙️ Actions")

if st.sidebar.button(
    "🗑️ Clear Document",
    use_container_width=True
):

    st.session_state.chunks = None
    st.session_state.vectorizer = None
    st.session_state.chunk_vectors = None
    st.session_state.document_name = None
    st.session_state.history = []

    st.rerun()


# --------------------------------------------------
# DOCUMENT UPLOAD
# --------------------------------------------------

st.subheader("📄 1. Upload Your Research Document")

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"]
)

if uploaded_file is not None:

    if st.button(
        "⚡ Process Document",
        use_container_width=True
    ):

        temp_path = None

        try:

            with st.spinner("Reading and processing document..."):

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as temp_file:

                    temp_file.write(uploaded_file.getvalue())
                    temp_path = temp_file.name

                loader = PyPDFLoader(temp_path)
                documents = loader.load()

                if not documents:
                    raise ValueError(
                        "No readable text found in this PDF."
                    )

                chunks = split_documents(documents)

                if not chunks:
                    raise ValueError(
                        "Could not create document chunks."
                    )

                vectorizer = TfidfVectorizer(
                    stop_words="english"
                )

                chunk_vectors = vectorizer.fit_transform(
                    [chunk.page_content for chunk in chunks]
                )

                st.session_state.chunks = chunks
                st.session_state.vectorizer = vectorizer
                st.session_state.chunk_vectors = chunk_vectors
                st.session_state.document_name = uploaded_file.name
                st.session_state.history = []

            st.success("Document processed successfully!")

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Pages Loaded", len(documents))

            with col2:
                st.metric("Chunks Created", len(chunks))

        except Exception as error:

            st.error(
                f"Document processing failed: {error}"
            )

        finally:

            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)


# --------------------------------------------------
# QUESTION AND ANSWER
# --------------------------------------------------

if st.session_state.chunks is not None:

    st.divider()

    st.subheader("💬 2. Ask Questions About Your Document")

    st.markdown(
        "Enter your question in the box below. "
        "You can ask multiple different questions about the same PDF."
    )

    question = st.text_area(
        "✍️ Your Question",
        placeholder="Example: What are the main findings of this research paper?",
        key="current_question"
    )

    if st.button(
        "🚀 Generate Answer",
        use_container_width=True
    ):

        if not question.strip():

            st.warning("Please enter a question.")

        else:

            try:

                with st.spinner(
                    "Retrieving relevant information and asking Gemini..."
                ):

                    question_vector = (
                        st.session_state.vectorizer.transform(
                            [question]
                        )
                    )

                    similarities = cosine_similarity(
                        question_vector,
                        st.session_state.chunk_vectors
                    ).flatten()

                    top_indices = similarities.argsort()[-5:][::-1]

                    relevant_docs = [
                        st.session_state.chunks[i]
                        for i in top_indices
                        if similarities[i] > 0
                    ]

                    if not relevant_docs:

                        answer = (
                            "I could not find relevant information "
                            "for this question in the uploaded document."
                        )

                        references = []

                    else:

                        answer = generate_answer_from_documents(
                            question,
                            relevant_docs
                        )

                        references = []
                        seen = set()

                        for doc in relevant_docs:

                            page = doc.metadata.get("page", None)

                            if page is not None:

                                page_number = int(page) + 1

                                if page_number not in seen:

                                    seen.add(page_number)
                                    references.append(page_number)

                st.session_state.history.append({
                    "question": question,
                    "answer": answer,
                    "references": references
                })

                st.rerun()

            except Exception as error:

                st.error(
                    f"Unable to generate answer: {error}"
                )


# --------------------------------------------------
# DISPLAY QUESTION HISTORY
# --------------------------------------------------

if st.session_state.history:

    st.divider()

    st.subheader("📖 3. Questions and Answers")

    for i, item in enumerate(
        reversed(st.session_state.history),
        start=1
    ):

        with st.expander(
            f"❓ Question {len(st.session_state.history) - i + 1}: "
            f"{item['question']}",
            expanded=(i == 1)
        ):

            st.markdown("### ❓ Your Question")
            st.info(item["question"])

            st.markdown("### 🤖 Generated Answer")
            st.markdown(item["answer"])

            if item["references"]:

                st.markdown("### 📌 Source Pages")

                st.write(
                    ", ".join(
                        str(page)
                        for page in item["references"]
                    )
                )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.markdown(
    '<div class="footer">'
    'PDF RAG Assistant | Python • Streamlit • TF-IDF • Google Gemini'
    '</div>',
    unsafe_allow_html=True
)
