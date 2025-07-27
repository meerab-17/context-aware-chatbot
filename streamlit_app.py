import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
import tempfile
import os

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

st.set_page_config(page_title="LangChain Chatbot", layout="wide")
st.title(" Context-Aware Chatbot with LangChain")

uploaded_file = st.file_uploader("📄 Upload a .txt file", type=["txt"])

if uploaded_file:
    with st.spinner(" Reading & Embedding..."):
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_path = temp_file.name

        loader = TextLoader(temp_path, encoding="utf-8")
        documents = loader.load()
        splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        docs = splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(docs, embeddings)

        llm = HuggingFaceHub(
            repo_id="google/flan-t5-small",
            model_kwargs={"temperature": 0.3, "max_length": 100}
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever()
        )

        st.success(" Ready! Ask your question:")
        query = st.text_input("Ask something based on the file:")
        if query:
            with st.spinner("Thinking..."):
                result = qa_chain.invoke({"query": query})
                st.write("🤖 **Answer:**", result["result"])
