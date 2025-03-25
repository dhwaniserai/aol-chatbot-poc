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
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
# Fetch values from .env file
CONNECTION_STRING = os.getenv("DATABASE_URL")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
MODEL_NAME = os.getenv("MODEL_NAME")

st.set_page_config(page_title="Art of Living Assistant")


def generate_llm_description(row):
    title = row["Project title"]

    prompt = f"""
You are an assistant that generates 4–5 sentence warm descriptions of Art of Living projects based on what you already know for the given project title:

Title: {title}
"""

    response = chat(model=MODEL_NAME, messages=[
        {"role": "system", "content": "You write informative and inspiring blurbs for service projects."},
        {"role": "user", "content": prompt}
    ])
    return response["message"]["content"]


# Custom Embedding Class for HuggingFace Model
class NomicEmbeddings(Embeddings):
    def __init__(self):
        self.model_name = "nomic-ai/nomic-embed-text-v1.5"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True)
        self.max_length = 8192

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        texts = [str(t) for t in texts]  # Ensure all inputs are strings
        encoded_input = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        with torch.no_grad():
            outputs = self.model(**encoded_input)
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

@st.cache_resource
def load_embeddings():
    return NomicEmbeddings()

embedding_model = load_embeddings()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=8000,
    chunk_overlap=200,
    length_function=lambda x: len(embedding_model.tokenizer.encode(str(x), add_special_tokens=False)),
    separators=["\n\n", "\n", ".", " ", ""]
)

# Load data
xls = pd.ExcelFile("Data Sample for Altro AI.xlsx")
sample_data = pd.read_excel(xls, "REAL and Mocked up Data for POC").fillna("")

descriptions = []
for _, row in sample_data.iterrows():
    try:
        desc = generate_llm_description(row)
        descriptions.append(desc)
    except Exception as e:
        print(f"Error on row {_}: {e}")
        descriptions.append("")

sample_data["Generated Description"] = descriptions
# Create documents from metadata
documents = []
for _, row in sample_data.iterrows():
    def safe(field): 
        value = row.get(field, "")
        if pd.isna(value):
            return ""
        if isinstance(value, pd.Timestamp):
            return value.strftime("%Y-%m-%d")
        return str(value)

    metadata = {
        "title": safe("Project title"),
        "link": safe("Links/Sources"),
        "locations": safe("Project Locations"),
        "target_group": safe("Target group"),
        "persona": safe("Persona"),
        "contact_person": safe("Contact Person"),
        "contact_title": safe("Contact Person Title"),
        "contact_email": safe("Contact Person email"),
        "volunteer_needs": safe("Volunteer Needs"),
        "volunteer_need_by": safe("Volunteer Need by (mm/dd/yyyy)"),
        "tenure": safe("Tenure"),
        "responsibilities": safe("Responsbilities"),
        "donation_needs": safe("Donation Needs"),
        "donation_amount": safe("Donation Amount ($)"),
        "donation_need_by": safe("Donation Need by"),
    }

    synthetic_description = row.get("Generated Description", "")


    if isinstance(synthetic_description, str) and synthetic_description.strip():
        documents.append(Document(
            page_content=synthetic_description,
            metadata=metadata
        ))

@st.cache_resource
def init_vectorstore():
    clean_documents = [
        Document(page_content=str(doc.page_content), metadata=doc.metadata)
        for doc in documents if isinstance(doc.page_content, str) and doc.page_content.strip()
    ]
    return PGVector.from_documents(
        documents=clean_documents,
        embedding=embedding_model,
        collection_name=COLLECTION_NAME,
        connection_string=CONNECTION_STRING,
        pre_delete_collection=True
    )

vector_store = init_vectorstore()

def semantic_search(query, top_k=3):
    return vector_store.similarity_search(query, k=top_k)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Welcome to Art of Living! How may I assist you today?"}]

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "Welcome to Art of Living! How may I assist you today?"}]

st.sidebar.button('ALTRO AI')
st.session_state["model"] = MODEL_NAME

def model_res_generator(user_input):
    relevant_projects = semantic_search(user_input)

    projects_info = "\n\n".join([
        f"**{doc.metadata['title']}**\n"
        f"- {doc.page_content}" for doc in relevant_projects
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