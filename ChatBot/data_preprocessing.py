import os
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# ========== SETTINGS ==========
GOOGLE_API_KEY = "YOUR API KEY"
EMBEDDINGS_MODEL = "models/embedding-001"
CHROMA_DB_DIR = "chroma_db"
TXT_FILE_PATH = "DICE_Info.txt"
# ===============================

def load_text_from_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return text
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None

def split_paragraphs(text):
    paragraphs = text.strip().split("\n\n")
    return [p.replace("\n", " ") for p in paragraphs if p.strip()]

def create_chroma_db(texts, db_dir, embeddings):
    print("✅ Creating new ChromaDB...")
    vector_db = Chroma.from_texts(
        texts,
        embeddings,
        persist_directory=os.path.join(db_dir, "myv-db.chroma")
    )
    print("✅ ChromaDB created and persisted!")

def main():
    print("🔧 Preprocessing started...")

    file_content = load_text_from_file(TXT_FILE_PATH)
    if file_content is None:
        print("⚠️ Exiting preprocessing due to missing file.")
        return
    
    texts = split_paragraphs(file_content)

    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDINGS_MODEL,
        google_api_key=GOOGLE_API_KEY
    )

    create_chroma_db(texts, CHROMA_DB_DIR, embeddings)

if __name__ == "__main__":
    main()
