import streamlit as st
import os, time
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS


load_dotenv()

## Load NVIDIA API KEY
os.environ['NVIDIA_API_KEY']=os.getenv('NVIDIA_API_KEY')

## LLM
llm=ChatNVIDIA(model="meta/llama-3.3-70b-instruct")

# Embedding
def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = NVIDIAEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader("./data")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:30])
        st.session_state.vectors = FAISS.from_documents(documents=st.session_state.final_documents, embedding=st.session_state.embeddings)

# UI
st.title("GoLang Q&A with NVIDIA NIM")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    </context>
    Question: {input}
"""
)

prompt1 = st.text_input("Enter your question from documents")

if st.button("Load Golang Brain"):
    vector_embedding()
    st.write("Golang NVIDIA FAISS Vector store is ready!")

if prompt1:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retriever_chain = create_retrieval_chain(retriever, document_chain)
    start = time.process_time()
    response = retriever_chain.invoke({'input': prompt1})
    print("Response Time : ", time.process_time() - start)
    st.write(response['answer'])

