import chromadb
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

DB_PATH = "./db"

# Embeddings (lokal)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Reranker (entscheidend für Qualität)
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection("docs")


def get_all_docs():
    data = collection.get(include=["documents"])
    return data["documents"]


def hybrid_search(query, top_k=10):

    # -------------------------
    # 1. VECTOR SEARCH
    # -------------------------
    q_emb = embedder.encode(query).tolist()

    vec = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        include=["documents"]
    )

    vec_docs = vec["documents"][0]

    # -------------------------
    # 2. BM25 SEARCH
    # -------------------------
    all_docs = get_all_docs()
    tokenized = [d.lower().split() for d in all_docs]

    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(query.lower().split())

    top_idx = np.argsort(scores)[::-1][:top_k]
    bm25_docs = [all_docs[i] for i in top_idx]

    # -------------------------
    # 3. MERGE
    # -------------------------
    candidates = list(set(vec_docs + bm25_docs))

    # -------------------------
    # 4. RERANKING
    # -------------------------
    pairs = [(query, doc) for doc in candidates]
    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(candidates, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return ranked[:3]


def ask():
    while True:
        q = input("\nFrage: ")

        results = hybrid_search(q)

        print("\n--- ANTWORT ---\n")

        for doc, score in results:
            print(doc)
            print("\n(score:", round(score, 3), ")\n---\n")


if __name__ == "__main__":
    ask()
