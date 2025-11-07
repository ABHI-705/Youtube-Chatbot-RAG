# 🎥 YouTube Chatbot – RAG Application (Talk to Your Videos)

## 🧠 Project Overview
This project is an AI-powered **YouTube Chatbot** built using **LangChain**, **Hugging Face**, and **Google Gemini**.  
You simply paste the **link to any YouTube video** (in Hindi or English), and the app automatically creates a **Retrieval-Augmented Generation (RAG)** system for that video.  

Once built, you can **chat directly with the video content** — ask questions, clarify points you didn’t understand, and explore topics in an interactive way.

For example:
> “What was the main idea explained at 5:45?”  
> “Summarize the conclusion part.”  
> “Explain the key takeaway from this tutorial.”

---

## 🚀 Features
✅ Accepts **YouTube video links** (Hindi or English)  
✅ Automatically **fetches and processes transcripts**  
✅ Builds a **RAG pipeline** using **LangChain + FAISS Vector Store**  
✅ Allows you to **chat with your video** in natural language  
✅ Uses **Hugging Face** or **Google Gemini LLM** for intelligent responses  
✅ Simple, clean **Streamlit UI**

---

## 🧩 Tech Stack
- **Python 3.10+**
- **Streamlit** – Web Interface  
- **LangChain** – RAG Framework  
- **Hugging Face Transformers** – Embeddings + Model Inference  
- **FAISS** – Vector Search for Context Retrieval  
- **YouTube Transcript API** – Fetches subtitles/transcripts  
- **Google Gemini API** – Language model for chat  
- **dotenv** – Secure API key management  

---

## 📦 Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/<your-username>/youtube-chatbot-rag.git
cd youtube-chatbot-rag
