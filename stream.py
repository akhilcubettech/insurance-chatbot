import os
from uuid import uuid4 as uid
import streamlit as st
from dotenv import load_dotenv
from langchain.schema.runnable.base import RunnableMap
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_parse import LlamaParse
from streamlit import session_state
import chromadb


load_dotenv()

model_name = "BAAI/bge-small-en"
model_kwargs = {"device": 'cpu'}
encode_kwargs = {"normalize_embeddings": True}
cache = "./cache/embed"

embedding_fn = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
    cache_folder=cache
)

parser = LlamaParse(
    result_type="markdown",
    premium_mode=True
)

llm = ChatMistralAI(
    model="mistral-large-latest",
    temperature=0,
    max_retries=2
)

LANGCHAIN_TRACING_V2 = os.getenv('LANGCHAIN_TRACING_V2')
LANGCHAIN_ENDPOINT = os.getenv('LANGCHAIN_ENDPOINT')
LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')
LANGCHAIN_PROJECT = os.getenv('LANGCHAIN_PROJECT')

chroma = chromadb.PersistentClient(path="./chromadb")

if "session_id" not in session_state:
    session_state["session_id"] = str(uid())

if "page" not in session_state:
    session_state["page"] = "upload"


def generate_text_embeddings(docs: str):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=256)
    chunks = text_splitter.split_text(docs)
    embeddings = embedding_fn.embed_documents(texts=chunks)
    return chunks, embeddings


def save_embeddings(session_id: str, docs: str):
    chunks, embeddings = generate_text_embeddings(docs)
    collection = chroma.get_or_create_collection(
        name=session_id,
        metadata={"hnsw:space": "cosine"}
    )
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        chunk_id = f"{session_id}_{i}"
        collection.upsert(
            ids=[chunk_id],
            embeddings=[embedding],
            metadatas=[{"session_id": session_id}],
            documents=[chunk]
        )
    return True


def process_policy_upload(file, session_id):
    docs = parser.load_data(file_path=file.read(), extra_info={'file_name': file.name})
    saved = save_embeddings(session_id, docs[0].text)
    return saved


def generate_answer(question: str, session_id: str):
    context = Chroma(client=chroma,
                     embedding_function=embedding_fn,
                     collection_name=session_id)
    history = session_state.messages
    print(history)
    template = """You are a knowledgeable and expert policy advisor named Whizz. 
    Your responses should be thoughtful and focused solely on questions related to Insurance policy. 
    If a question falls outside the scope of policy or the information available, kindly respond with, 
    'I'm sorry, but I can only assist with questions related to Insurance Policy.' 
    Refrain from providing information or insights on unrelated topics, such as sports or general knowledge. 
    You are provided with relevant context and chat history:
    {context} {history}

    Question: {question}
    """
    msg = ChatPromptTemplate.from_template(template)
    output_parser = StrOutputParser()
    retriever = context.as_retriever(search_type="mmr", k=10,
                                     search_kwargs={'k': 10, "score_threshold": 0.5, 'fetch_k': 50})
    chain = RunnableMap({
        "context": lambda x: retriever.invoke(x["question"]),
        "question": lambda x: x["question"],
        "history": lambda x: history
    }) | msg | llm | output_parser

    response_chunks = []

    def stream_and_collect():
        for ck in chain.stream({"question": question}):
            response_chunks.append(ck)
            yield ck

    return stream_and_collect()



if session_state["page"] == "upload":
    st.title("Upload Your Insurance Policy Document")
    uploaded_file = st.file_uploader("Upload policy document here", type=["pdf"])
    file_button_pressed = st.button("Upload")
    if file_button_pressed and uploaded_file:
        with st.spinner("Processing uploaded file..."):
            uploaded = process_policy_upload(file=uploaded_file, session_id=session_state.session_id)
            if uploaded:
                session_state['status'] = "True"
                st.success("File uploaded and processed successfully!")
                session_state["page"] = "chat"  # Switch to chat page after successful upload
                st.rerun()
            else:
                st.error("Error uploading file!")


elif session_state["page"] == "chat":
    st.title("Chat with Insurance Assistant")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("How can i help you ?"):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response = st.write_stream(generate_answer(prompt, session_state.session_id))

        st.session_state.messages.append({"role": "assistant", "content": response})

