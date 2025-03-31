import os
from dotenv import load_dotenv
from app.data_loader import load_documents
from app.vector_store import get_vectorstore
from app.config import DATABASE_URL

def transfer_data():
    # Load documents from Excel
    print("Loading documents from Excel...")
    documents = load_documents()
    
    # Initialize vector store with Neon database
    print("Initializing vector store with Neon database...")
    vectorstore = get_vectorstore()
    
    print("Data transfer complete!")

if __name__ == "__main__":
    load_dotenv()
    transfer_data() 