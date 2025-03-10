import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from ollama import chat
import torch
from transformers import AutoModel, AutoTokenizer
from langchain.vectorstores.pgvector import PGVector
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# Fetch values from .env file
CONNECTION_STRING = os.getenv("DATABASE_URL")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
MODEL_NAME = os.getenv("MODEL_NAME")

st.set_page_config(page_title="Art of Living Assistant")

# Custom Embedding Class for HuggingFace Model
class NomicEmbeddings(Embeddings):
    def __init__(self):
        self.model_name = "nomic-ai/nomic-embed-text-v1.5"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True)
        self.max_length = 8192

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts using mean pooling"""
        encoded_input = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        with torch.no_grad():
            outputs = self.model(**encoded_input)
        
        # Mean pooling
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query"""
        return self.embed_documents([text])[0]
    
@st.cache_resource
def load_embeddings():
    return NomicEmbeddings()

embedding_model = load_embeddings()


# Text splitting configuration
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=8000,
    chunk_overlap=200,
    length_function=lambda x: len(embedding_model.tokenizer.encode(x, add_special_tokens=False)),
    separators=["\n\n", "\n", ".", " ", ""]
)

# Load and process data
xls = pd.ExcelFile("Data Sample for Altro AI.xlsx")
sample_data = pd.read_excel(xls, "Sample").fillna("")

# Create documents with chunking
documents = []
for _, row in sample_data.iterrows():
    chunks = text_splitter.split_text(row["Description"])
    for chunk in chunks:
        documents.append(Document(
            page_content=chunk,
            metadata={
                "title": row["Project title"],
                "locations": row["Locations"],
                "funding_needed": row["Project Needs"],
                "volunteers_needed": row["Volunteer Needs"],
                "impact_focus": row["Impact focus"],
                "donation_link": row["Donation link"]
            }
        ))

# Create vector store with custom embeddings
@st.cache_resource
def init_vectorstore():
    return PGVector.from_documents(
        documents=documents,
        embedding=embedding_model,
        collection_name=COLLECTION_NAME,
        connection_string=CONNECTION_STRING,
        pre_delete_collection=True  # Optional: Clear existing data
    )

vector_store = init_vectorstore()

def semantic_search(query, top_k=3):
    return vector_store.similarity_search(query, k=top_k)

# Session state management
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Welcome to Art of Living! How may I assist you today?"}]

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "Welcome to Art of Living! How may I assist you today?"}]

st.sidebar.button('ALTRO AI')
st.session_state["model"] = MODEL_NAME

def model_res_generator(user_input):
    # Perform semantic search
    relevant_projects = semantic_search(user_input)
    
    # Format results for LLM
    projects_info = "\n\n".join([
        f"**{doc.metadata['title']}**\n"
        f"- Description: {doc.page_content}\n"
        f"- Locations: {doc.metadata['locations']}\n"
        f"- Funding Needed: {doc.metadata['funding_needed']}\n"
        f"- Volunteers Needed: {doc.metadata['volunteers_needed']}\n"
        f"- Impact Focus: {doc.metadata['impact_focus']}\n"
        f"- Donation Link: {doc.metadata['donation_link']}"
        for doc in relevant_projects
    ])
    
    # Generate response using Ollama
    stream = chat(
        model=st.session_state["model"],
        messages=[
            {"role": "system", "content": f"""
You are an AI assistant for the **Art of Living**, dedicated to spreading peace, well-being, and service.  
Your mission is to engage users warmly, inspire them, and encourage meaningful contributions through donations, volunteering, or participation in Art of Living's transformative initiatives.

### **INSTRUCTIONS**  
1️⃣ ALWAYS recommend SPECIFIC projects from the database.  
2️⃣ Match based on location, interests, and budget constraints.  
3️⃣ For **donations**, show only projects within the budget.  
4️⃣ If no exact match, suggest the closest projects with an explanation.  
5️⃣ For **volunteering**, match based on location & required skills.  
6️⃣ NEVER invent projects beyond the database.  

### **Relevant Project Information:**  
{projects_info}
"""},

            {"role": "user", "content": user_input},
        ],
        stream=True,
    )
    for chunk in stream:
        yield chunk["message"]["content"]

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about projects, donations, or volunteering..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        response = st.write_stream(model_res_generator(prompt))
        st.session_state.messages.append({"role": "assistant", "content": response})
        
# Can implement filter based search in model_res_generator for better search