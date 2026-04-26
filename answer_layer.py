import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

DB_PATH = "./db"

# Embedding + Reranker
embedder = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection("docs")


# -------------------------
# RETRIEVAL
# -------------------------
def retrieve(query, top_k=8):

    q_emb = embedder.encode(query).tolist()

    res = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        include=["documents"]
    )

    docs = res["documents"][0]

    pairs = [(query, d) for d in docs]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

    return ranked[:3]


# -------------------------
# ANSWER BUILDER
# -------------------------
def build_answer(query, passages):

    query_words = set(query.lower().split())
    answers = []

    for doc, score in passages:
        sentences = doc.split(". ")

        for s in sentences:
            s_low = s.lower()

            # Relevanz-Regel:
            # 1. Keyword match ODER
            # 2. enthält Zahlen (wichtig für Gesetz/Maut)
            if (
                any(w in s_low for w in query_words)
                or any(char.isdigit() for char in s)
            ):
                answers.append(s.strip())

    # Fallback, falls nichts gefunden wird
    if not answers:
        answers = [passages[0][0]]

    # Duplikate entfernen
    seen = list(dict.fromkeys(answers))

    return seen[:4]


# -------------------------
# MAIN LOOP
# -------------------------
def run():

    while True:
        q = input("\nFrage (exit zum Beenden): ")

        if q.lower() in ["exit", "quit"]:
            break

        passages = retrieve(q)
        answer = build_answer(q, passages)

        print("\n--- ANTWORT ---\n")
        for a in answer:
            print("•", a)

        print("\n--- QUELLEN (Scores) ---\n")
        for doc, score in passages:
            print(f"[{round(score,3)}]")
            print(doc[:200])
            print("---")


if __name__ == "__main__":
    run()
