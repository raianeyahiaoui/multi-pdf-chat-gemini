# multi-pdf-chat-gemini
Chat with multiple PDF documents using LangChain and Google Gemini Pro (RAG-based system).
# 📚 Multi-PDF Chat with Gemini Pro

Chat with multiple PDF documents using **LangChain**, **FAISS**, and **Google Gemini Pro**.  
Upload your PDFs, ask questions, and get contextual answers powered by RAG (Retrieval Augmented Generation).

---

## 🚀 Problem
Reading and searching large PDF collections is time-consuming. Traditional keyword search fails to understand context.

## 💡 Approach
- Extract text from multiple PDFs.
- Split into semantic chunks.
- Store embeddings in FAISS vector database.
- Use Gemini Pro to generate contextual answers from retrieved chunks.

## ✨ Innovation
- Combines **LangChain + Gemini Pro** for intelligent multi-PDF Q&A.
- Retrieval-Augmented Generation (RAG) pipeline.
- Easy-to-use Streamlit interface.

---

## 🛠️ Tech Stack
- Python
- LangChain
- FAISS
- HuggingFace Embeddings
- Google Gemini Pro
- Streamlit

---

## ▶️ Run Locally

```bash
git clone https://github.com/raianeyahiaoui/multi-pdf-chat-gemini.git
cd multi-pdf-chat-gemini
pip install -r requirements.txt
streamlit run app.py
