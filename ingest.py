import os
import uuid
from pypdf import PdfReader
import chromadb
from sentence_transformers import SentenceTransformer

DATA_PATH = "./data"
DB_PATH = "./db"

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection("docs")


def load_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t + "\n"
    return text


def chunk(text, size=400, overlap=80):
    words = text.split()
    chunks = []

    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i:i+size]))
        i += size - overlap

    return chunks


def ingest_file(file_path):
    text = load_pdf(file_path)
    chunks = chunk(text)

    for c in chunks:
        emb = model.encode(c).tolist()

        collection.add(
            ids=[str(uuid.uuid4())],
            embeddings=[emb],
            documents=[c],
            metadatas=[{"source": os.path.basename(file_path)}]
        )

    print(f"✔ indexed {file_path}")


for file in os.listdir(DATA_PATH):
    if file.endswith(".pdf"):
        ingest_file(os.path.join(DATA_PATH, file))

print("DONE")

