from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from gtts import gTTS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from tempfile import NamedTemporaryFile
from langchain_astradb import AstraDBVectorStore
from dotenv import load_dotenv
import speech_recognition as sr
import fitz  # PyMuPDF to extract text from PDFs
import numpy as np
import base64
import pyttsx3
from time import sleep
import uuid

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Configure embeddings and vector store
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")

# Configure embeddings and vector stores
embeddings_model = OpenAIEmbeddings()
local_embeddings_path = "resume_embeddings.json"

# Configure AstraDB vector store
vstore = AstraDBVectorStore(
    collection_name="test",
    embedding=embeddings_model,
    token=ASTRA_DB_APPLICATION_TOKEN,
    api_endpoint=ASTRA_DB_API_ENDPOINT,
)

retriever_astra = vstore.as_retriever(search_kwargs={"k": 3})

# Utility functions
# def speak(text):
#     for word in text.split():
#         if not st.session_state["mute"]:
#             engine = pyttsx3.init()
#             engine.setProperty("volume", 1.0)
#             engine.setProperty("rate", 150)
#             engine.say(text)
#             engine.runAndWait()
#         else:
#             engine.setProperty("volume", 0.0)

def extract_text_from_pdf(pdf_file):
    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_file.getvalue())
        tmp_path = tmp.name
    try:
        doc = fitz.open(tmp_path)
        resume_text = ""
        for page in doc:
            resume_text += page.get_text()
        return resume_text
    except Exception as e:
        # st.error(f"Error extracting text from PDF: {e}")
        return ""

def generate_resume_embedding(resume_text):
    return embeddings_model.embed_query(resume_text)

def save_embedding_to_astra(embedding_data):
    """Save the embedding to AstraDB."""
    document = Document(
        id=str(hash(embedding_data["content"])),
        page_content=embedding_data["content"],
        metadata=embedding_data["metadata"]
    )
    vstore.add_documents([document])

def encode_embedding_to_base64(embedding):
    return base64.b64encode(np.array(embedding, dtype=np.float32).tobytes()).decode("utf-8")


retriever = vstore.as_retriever(search_kwargs={"k": 3})
prompt_template="""
    use the following context to answer the question asked.
    please try to provide the answer only based on the context and format the answer in markdown format
    context: {context}
    question: {question}
    answers:
    """
prompt=PromptTemplate(template=prompt_template, input_variables=["context", "question"])

#document_chain = create_stuff_documents_chain(llm, prompt)

qa_chain = RetrievalQA.from_chain_type(
    retriever= retriever_astra,
    chain_type="stuff",
    llm=ChatGroq(api_key=groq_api_key, model="llama-3.1-8b-instant"),
    chain_type_kwargs={"prompt":prompt}
)

@app.route("/upload_resume", methods=["POST"])
def upload_resume():
    file = request.files.get("file")
    if not file or not file.filename.endswith(".pdf"):
        return jsonify({"error": "Invalid file format. Only PDF is allowed."}), 400

    try:
        # doc = fitz.open(stream=file.read(), filetype="pdf")
        # resume_text = ""
        # for page in doc:
        #     resume_text += page.get_text()
        resume_content = extract_text_from_pdf(file)
        resume_embedding = embeddings_model.embed_query(resume_content)
        encoded_embedding = encode_embedding_to_base64(resume_embedding)

        embedding_data = {
            "content": resume_content,
            "embedding": encoded_embedding,
            "metadata": {
                "source": "resume",
                "id": str(uuid.uuid4()),
            }
        }
        save_embedding_to_astra(embedding_data)

        return jsonify({"message": "Resume uploaded and processed successfully!"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/query", methods=["POST"])
def query_job():
    data = request.json
    query = data.get("query")
    if not query:
        return jsonify({"error": "Query cannot be empty."}), 400

    try:
        response = qa_chain.run({"query": query})
        return jsonify({"result": response}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
