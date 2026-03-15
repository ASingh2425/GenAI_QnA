# GenAI Document Question Answering System
A Generative AI application that allows users to upload a PDF document and ask questions about its content.  
The system uses a **Retrieval-Augmented Generation (RAG)** pipeline combining semantic search with a language model to generate context-aware answers.

## Features
- Upload and analyze PDF documents
- Ask multiple questions about the document
- Semantic search using vector embeddings
- Conversational interface built with Streamlit
- Fast retrieval using FAISS vector database

## Tech Stack
- Python
- Streamlit
- LangChain
- FAISS
- HuggingFace Transformers
- Sentence Transformers

## Architecture
PDF Upload → Text Extraction → Chunking → Embeddings → FAISS Vector Search → LLM → Answer Generation

## Installation
Clone the repository:
git clone https://github.com/ASingh2425/genai-document-qa.git

## Install dependencies:
pip install -r requirements.txt

## Run the App
streamlit run app.py

## Example Questions
- What is this document about?
- Summarize the key points.
- What architecture or methodology is described?

## Author
**Anvesha Singh**  
B.Tech – Data Science & Engineering  
Manipal Institute of Technology

GitHub: https://github.com/ASingh2425  
LinkedIn: https://linkedin.com/in/anvesha-singh-3427202aa
