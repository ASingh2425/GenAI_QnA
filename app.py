import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from transformers import pipeline

st.set_page_config(page_title="GenAI Document QA", layout="wide")

st.title("GenAI Document Question Answering System")

# ---- SESSION STATE ----
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---- FILE UPLOAD ----
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:

    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings()

    st.session_state.vectorstore = FAISS.from_documents(chunks, embeddings)

    st.success("Document processed! Ask questions below.")

# ---- QUESTION INPUT ----
question = st.text_input("Ask a question about the document")

# ---- LOAD MODEL (ONLY ONCE) ----
@st.cache_resource
def load_model():
    return pipeline(
        "text-generation",
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )

qa_model = load_model()

# ---- QUESTION ANSWERING ----
if question and st.session_state.vectorstore:

    docs = st.session_state.vectorstore.similarity_search(question, k=4)

    context = " ".join([doc.page_content for doc in docs])

    context = context[:1200]

    prompt = f"""
Answer the question in 2-3 sentences using the context.

Context:
{context}

Question: {question}

Answer:
"""

    result = qa_model(
        prompt,
        max_new_tokens=120,
        do_sample=False
    )

    answer = result[0]["generated_text"].replace(prompt, "").strip()

    # Save conversation
    st.session_state.chat_history.append(("User", question))
    st.session_state.chat_history.append(("AI", answer))

# ---- DISPLAY CHAT HISTORY ----
st.subheader("Chat")

for role, text in st.session_state.chat_history:
    if role == "User":
        st.markdown(f"**You:** {text}")
    else:
        st.markdown(f"**AI:** {text}")
