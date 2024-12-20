import os
import pdfplumber
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
import streamlit as st
from typing import Dict, List, Any


class RAGPipeline:
    def __init__(self, api_key: str = None, model_name: str = 'all-MiniLM-L6-v2', embedding_dim: int = 384):
        """
        Initialize the RAG pipeline with necessary components.
        
        Args:
            api_key (str): OpenAI API key. If None, will try to get from environment variable.
            model_name (str): Name of the sentence transformer model.
            embedding_dim (int): Dimension of the embeddings.
        """
        self.model = SentenceTransformer(model_name)
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.chunk_metadata = []
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len
        )

        # Handle API key with multiple fallbacks
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            st.error("OpenAI API key not found. Please provide it in the sidebar.")
            return

        # Initialize OpenAI LLM
        try:
            self.llm = OpenAI(
                model_name="gpt-3.5-turbo",
                openai_api_key=self.api_key  # Correctly pass the API key
            )
        except Exception as e:
            st.error(f"Error initializing OpenAI: {str(e)}")
            return

        self.retrieval_prompt = PromptTemplate(
            template="Answer the user's question based on the following information:\n{context}\n\nQuestion: {question}\nAnswer:",
            input_variables=["context", "question"]
        )

    def extract_text_from_pdf(self, pdf_file) -> Dict[int, str]:
        """Extract text from PDF file."""
        try:
            text_data = {}
            with pdfplumber.open(pdf_file) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text()
                    if text:
                        text_data[page_num] = text
            return text_data
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")

    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks."""
        if not isinstance(text, str):
            raise TypeError(f"Expected string input, got {type(text)}")
        return self.text_splitter.split_text(text)

    def process_documents(self, files: List[Any]) -> None:
        """Process multiple PDF documents and add to FAISS index."""
        all_text = ""
        for file in files:
            text_data = self.extract_text_from_pdf(file)
            all_text += " ".join(text_data.values())

        chunks = self.chunk_text(all_text)
        embeddings = self.model.encode(chunks)
        embeddings = np.array(embeddings, dtype=np.float32)
        self.index.add(embeddings)
        self.chunk_metadata.extend([{
            "chunk": chunk,
            "page": idx + 1
        } for idx, chunk in enumerate(chunks)])

    def search_documents(self, query: str, k: int = 5) -> str:
        """Search documents with the query and return response."""
        query_embedding = self.model.encode(query)
        query_embedding = np.array([query_embedding], dtype=np.float32)
        distances, indices = self.index.search(query_embedding, k)

        if len(indices[0]) == 0 or indices[0][0] == -1:
            return "No relevant information found for your query."

        retrieved_chunks = [self.chunk_metadata[idx]["chunk"] for idx in indices[0]]
        context = "\n".join(retrieved_chunks)

        try:
            response = self.llm(self.retrieval_prompt.format(
                context=context,
                question=query
            ))
            return response
        except Exception as e:
            return f"Error generating response: {str(e)}"


def main():
    st.title("Chat with PDF using RAG")

    # Add API key input to sidebar
    api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password")
    if not api_key:
        st.sidebar.warning("Please enter your OpenAI API key to continue")
        return

    # Initialize RAG pipeline with provided API key
    rag = RAGPipeline(api_key=api_key)

    # File upload
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        accept_multiple_files=True,
        type="pdf"
    )

    # Query input
    query = st.text_input("Enter your query")

    if st.button("Get Answer"):
        if not uploaded_files:
            st.error("Please upload at least one PDF file.")
            return

        if not query.strip():
            st.error("Please enter a query.")
            return

        try:
            # Process documents
            with st.spinner("Processing documents..."):
                rag.process_documents(uploaded_files)

            # Get response
            with st.spinner("Searching for answer..."):
                response = rag.search_documents(query)
                st.write(response)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
