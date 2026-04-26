import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

DB_PATH = "./db"

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
# CLEAN ANSWER ENGINE
# -------------------------
def generate_answer(query, passages):

    query_words = set(query.lower().split())
    extracted_sentences = []

    for doc, score in passages:
        sentences = doc.split(". ")

        for s in sentences:
            s_clean = s.strip()

            # Relevanz:
            if (
                any(w in s_clean.lower() for w in query_words)
                or any(char.isdigit() for char in s_clean)
                or "€" in s_clean
                or "maut" in s_clean.lower()
            ):
                extracted_sentences.append(s_clean)

    # Fallback
    if not extracted_sentences:
        extracted_sentences = [passages[0][0]]

    # Dedupe
    unique = list(dict.fromkeys(extracted_sentences))

    # 🧠 CLEAN OUTPUT (kein Rohtext mehr)
    answer = " ".join(unique[:4])

    return answer


# -------------------------
# MAIN LOOP
# -------------------------
def run():

    while True:
        q = input("\nFrage (exit zum Beenden): ")

        if q.lower() in ["exit", "quit"]:
            break

        passages = retrieve(q)
        answer = generate_answer(q, passages)

        print("\n--- ANTWORT ---\n")
        print(answer)

        print("\n--- QUELLEN ---\n")

        for doc, score in passages:
            print(f"[score: {round(score,3)}]")
            print(doc[:200])
            print("---")


if __name__ == "__main__":
    run()
