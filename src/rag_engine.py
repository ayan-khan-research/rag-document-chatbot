import os
import re
from typing import List, Tuple


class RAGEngine:
    """
    Core RAG engine supporting:
    - OpenAI GPT-4o / GPT-3.5 Turbo
    - Ollama (Llama3, Mistral) — local models
    - FAISS or ChromaDB vector store
    - OpenAI or HuggingFace embeddings
    """

    def __init__(
        self,
        texts: List[str],
        model_choice: str,
        embed_choice: str,
        vector_store: str,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        top_k: int = 5,
        temperature: float = 0.1,
    ):
        self.model_choice = model_choice
        self.embed_choice = embed_choice
        self.vector_store_choice = vector_store
        self.top_k = top_k
        self.temperature = temperature
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Build vector store
        self.vectorstore = self._build_vectorstore(texts)
        # Build LLM
        self.llm = self._build_llm()
        # Build QA chain
        self.qa_chain = self._build_qa_chain()

    # ── Build embeddings ────────────────────────────────────────────────────
    def _build_embeddings(self):
        if "OpenAI" in self.embed_choice:
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(model="text-embedding-3-small")

        elif "MiniLM" in self.embed_choice:
            from langchain_huggingface import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )
        else:
            from langchain_huggingface import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(
                model_name="BAAI/bge-small-en-v1.5",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )

    # ── Build vector store ──────────────────────────────────────────────────
    def _build_vectorstore(self, texts: List[str]):
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain.schema import Document

        # Split texts into smaller chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        docs = []
        for text in texts:
            # Extract source metadata from the [Source: ...] prefix
            source_match = re.match(r'\[Source: ([^\]]+)\]', text)
            source = source_match.group(1) if source_match else "Unknown"
            clean_text = re.sub(r'\[Source: [^\]]+\]\n?', '', text).strip()

            chunks = splitter.split_text(clean_text)
            for chunk in chunks:
                if chunk.strip():
                    docs.append(Document(
                        page_content=chunk,
                        metadata={"source": source}
                    ))

        embeddings = self._build_embeddings()

        if "FAISS" in self.vector_store_choice:
            from langchain_community.vectorstores import FAISS
            return FAISS.from_documents(docs, embeddings)
        else:
            from langchain_community.vectorstores import Chroma
            return Chroma.from_documents(docs, embeddings)

    # ── Build LLM ───────────────────────────────────────────────────────────
    def _build_llm(self):
        if "GPT-4o" in self.model_choice:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model="gpt-4o",
                temperature=self.temperature,
                max_tokens=2000
            )
        elif "GPT-3.5" in self.model_choice:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=self.temperature,
                max_tokens=1500
            )
        elif "Llama3" in self.model_choice:
            from langchain_ollama import ChatOllama
            return ChatOllama(
                model="llama3",
                temperature=self.temperature
            )
        else:
            from langchain_ollama import ChatOllama
            return ChatOllama(
                model="mistral",
                temperature=self.temperature
            )

    # ── Build QA chain ──────────────────────────────────────────────────────
    def _build_qa_chain(self):
        from langchain.chains import RetrievalQAWithSourcesChain
        from langchain.prompts import PromptTemplate

        template = """You are an expert document assistant. Answer questions ONLY based on the provided context.
If the answer is not in the context, say "I couldn't find this information in the uploaded documents."

Always:
- Be precise and accurate
- Quote relevant sections when helpful
- Mention which document/section the info comes from
- If multiple documents address the question, synthesize all relevant info

Context:
{summaries}

Question: {question}

Answer:"""

        prompt = PromptTemplate(
            template=template,
            input_variables=["summaries", "question"]
        )

        retriever = self.vectorstore.as_retriever(
            search_type="mmr",  # Maximum Marginal Relevance for diversity
            search_kwargs={
                "k": self.top_k,
                "fetch_k": self.top_k * 2
            }
        )

        return RetrievalQAWithSourcesChain.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )

    # ── Query ────────────────────────────────────────────────────────────────
    def query(self, question: str) -> Tuple[str, List[str], float]:
        """
        Returns (answer, list_of_sources, confidence_score)
        """
        try:
            result = self.qa_chain.invoke({"question": question})

            answer = result.get("answer", "No answer generated.")
            source_docs = result.get("source_documents", [])

            # Extract unique sources
            sources = list(dict.fromkeys([
                doc.metadata.get("source", "Unknown")
                for doc in source_docs
            ]))

            # Confidence: based on number of relevant chunks retrieved
            confidence = min(0.95, 0.5 + (len(source_docs) / (self.top_k * 2)))

            return answer, sources, confidence

        except Exception as e:
            error_msg = str(e)
            if "API key" in error_msg or "authentication" in error_msg.lower():
                return (
                    "❌ API key error. Please check your OpenAI API key in the sidebar.",
                    [],
                    0.0
                )
            elif "Connection" in error_msg or "ollama" in error_msg.lower():
                return (
                    "❌ Cannot connect to Ollama. Make sure Ollama is running locally with `ollama serve`.",
                    [],
                    0.0
                )
            else:
                return f"❌ Error: {error_msg}", [], 0.0
