# 📚 DocChat — Document-Based Chatbot using RAG

**CA2 Hackathon Project | Department of Information Technology | IT315E**
**KIET Group of Institutions**

---

## 🚀 What This Does

DocChat is an AI-powered chatbot that answers questions from your documents (PDF, DOCX, TXT, CSV, Excel) using **Retrieval-Augmented Generation (RAG)**. It retrieves the most relevant sections from your documents and uses an LLM to generate accurate, cited answers.

---

## 🏗️ Architecture

```
Documents → Text Extraction → Chunking → Embeddings → Vector Store (FAISS/ChromaDB)
                                                              ↑
User Query → Query Embedding → Similarity Search ────────────┘
                                     ↓
                            Top-K Relevant Chunks
                                     ↓
                          LLM (GPT-4o / Llama3 / Mistral)
                                     ↓
                        Answer + Source Citations
```

---

## 📦 Setup Instructions

### Step 1: Clone / Download the project
```bash
cd rag_chatbot
```

### Step 2: Create virtual environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### Step 3: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the app
```bash
streamlit run app.py
```

The app will open at: **http://localhost:8501**

---

## 🤖 Model Options

| Model | Cost | Setup | Accuracy |
|-------|------|-------|----------|
| OpenAI GPT-4o | Paid API | Just API key | ⭐⭐⭐⭐⭐ |
| OpenAI GPT-3.5 Turbo | Cheaper API | Just API key | ⭐⭐⭐⭐ |
| Ollama Llama3 (Local) | FREE | Install Ollama | ⭐⭐⭐⭐ |
| Ollama Mistral (Local) | FREE | Install Ollama | ⭐⭐⭐⭐ |

### Using Ollama (Free Local Models):
```bash
# Install from: https://ollama.com
ollama pull llama3
ollama serve
```

---

## 📊 Features

- **Multi-format Support**: PDF, DOCX, TXT, CSV, Excel
- **Source Citations**: Every answer shows which document/page it came from
- **Analytics Dashboard**: Query count, response time, confidence scores
- **Model Flexibility**: Switch between OpenAI & Ollama
- **Vector Store Choice**: FAISS (fast) or ChromaDB (persistent)
- **MMR Retrieval**: Maximum Marginal Relevance for diverse results
- **Configurable RAG**: Adjust chunk size, overlap, top-K from UI

---

## 🛠️ Technologies Used

| Technology | Purpose |
|-----------|---------|
| Streamlit | Web UI / Dashboard |
| LangChain | RAG orchestration |
| FAISS | Vector similarity search |
| ChromaDB | Persistent vector storage |
| HuggingFace Transformers | Free embeddings |
| OpenAI API | GPT-4o / GPT-3.5 |
| Ollama | Local LLM inference |
| pdfplumber | PDF text extraction |
| python-docx | Word document parsing |
| pandas | CSV/Excel processing |

---

## 📁 Project Structure

```
rag_chatbot/
├── app.py                  # Main Streamlit app
├── requirements.txt        # All dependencies
├── README.md               # This file
└── src/
    ├── __init__.py
    ├── document_processor.py   # Multi-format document parser
    ├── rag_engine.py           # Core RAG logic
    └── analytics.py            # Dashboard metrics tracker
```

---

## 🏆 Innovation Highlights

1. **Multi-model support** — Switch between GPT-4o, GPT-3.5, Llama3, Mistral from the UI
2. **Context-aware citations** — Every answer references the source document and page
3. **MMR Retrieval** — Avoids repetitive chunks, gets diverse relevant context
4. **Real-time Analytics** — Track performance metrics as you use the system
5. **Zero config for free tier** — Works with Ollama completely offline and free

---

## 👤 Developer

**Ayan Khan**  

*Independently developed for the IT315E Hackathon, focusing on practical applications of Generative AI.*
