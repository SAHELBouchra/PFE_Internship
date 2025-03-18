import os
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# ========== SETTINGS ==========
GOOGLE_API_KEY = "AIzaSyCbAphc7pmbA2Oo44Um9479jaJciALPUdQ"
EMBEDDINGS_MODEL = "models/embedding-001"
CHROMA_DB_DIR = "chroma_db"
# ===============================

def get_retriever(k: int = 2):
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDINGS_MODEL,
        google_api_key=GOOGLE_API_KEY
    )

    db_path = os.path.join(CHROMA_DB_DIR, "myv-db.chroma")

    if not os.path.exists(db_path):
        raise FileNotFoundError("❌ ChromaDB not found! Run data_preprocessing.py first.")

    print("✅ Loading ChromaDB retriever...")
    vector_db = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings
    )

    return vector_db.as_retriever(search_kwargs={"k": k})
