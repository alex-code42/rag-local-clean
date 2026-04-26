import chromadb
from sentence_transformers import SentenceTransformer

DB_PATH = "./db"

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection("docs")


def ask(q):
    emb = model.encode(q).tolist()

    res = collection.query(
        query_embeddings=[emb],
        n_results=3,
        include=["documents"]
    )

    docs = res["documents"][0]

    if not docs:
        return "Keine Treffer gefunden."

    return "\n\n---\n\n".join(docs)


while True:
    q = input("Frage: ")
    print("\nAntwort:\n")
    print(ask(q))
    print("\n")

