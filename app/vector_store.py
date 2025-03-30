from langchain.vectorstores.pgvector import PGVector
from app.embeddings import NomicEmbeddings
from app.data_loader import load_documents
from app.config import COLLECTION_NAME, DATABASE_URL

def init_vectorstore():
    documents = load_documents()
    embeddings = NomicEmbeddings()
    return PGVector.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        connection_string=DATABASE_URL,
        pre_delete_collection=True
    )
