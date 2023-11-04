import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.vectorstores.pgvector import PGVector
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

# Load environment variables from a .env file
load_dotenv()

# Streamlit header
st.header("GenAI Q&A with pgvector and Amazon Aurora PostgreSQL")

# User input for the question
user_question = st.chat_input("Ask a question about your documents:")

# Initialize the HuggingFaceHub model
llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature": 0.5, "max_length": 1024})


# Define a function to extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    print(text)
    return text


# Define a function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    chunks = text_splitter.split_text(text)
    print(chunks)
    return chunks


# Define the PostgreSQL connection string
CONNECTION_STRING = PGVector.connection_string_from_db_params(
    driver=os.getenv("PGVECTOR_DRIVER"),
    user=os.getenv("PGVECTOR_USER"),
    password=os.getenv("PGVECTOR_PASSWORD"),
    host=os.getenv("PGVECTOR_HOST"),
    port=os.getenv("PGVECTOR_PORT"),
    database=os.getenv("PGVECTOR_DATABASE")
)


# Define a function to create a vector store from text chunks
def get_vectorstore(text_chunks):
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    embeddings = HuggingFaceInstructEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs
    )
    vectorstore = PGVector.from_texts(texts=text_chunks, embedding=embeddings, connection_string=CONNECTION_STRING)
    print(vectorstore)
    return vectorstore


# Define a function to create a conversation chain
def get_conversation_chain(vectorstore):
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(),
                                                               memory=memory)
    return conversation_chain

#
# # Create a Streamlit session state to store the conversation chain
# if 'conversation' not in st.session_state:
#     st.session_state.conversation = get_conversation_chain(vectorstore)

# User input for the question (again)
if user_question:
    st.chat_input(user_question)

# Streamlit sidebar
with st.sidebar:
    st.subheader("Your documents")

    # File uploader for PDFs
    pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)

    if st.button("Process"):
        if pdf_docs:
            with st.spinner("Processing"):
                # Get PDF text
                raw_text = get_pdf_text(pdf_docs)

                # Get text chunks
                text_chunks = get_text_chunks(raw_text)

                # Create a vector store from text chunks
                vectorstore = get_vectorstore(text_chunks)

                # Update the conversation chain with the new vector store
                st.session_state.conversation = get_conversation_chain(vectorstore)
