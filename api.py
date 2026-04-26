from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb

app = FastAPI()
sessions = {}
DB_PATH = "./db"

# Models
embedder = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# DB Client
db_client = chromadb.PersistentClient(path=DB_PATH)
collection = db_client.get_collection("docs")


class Query(BaseModel):
    question: str

def get_user_id():
    return "default_user"

def retrieve(query):
    q_emb = embedder.encode(query).tolist()

    res = collection.query(
        query_embeddings=[q_emb],
        n_results=8,
        include=["documents"]
    )

    docs = res["documents"][0]
    
    print("QUERY:", query)
    print("TOP DOCS:", docs[:2])
   
    pairs = [(query, d) for d in docs]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    
    threshold = 0.25 if len(query.split()) > 5 else 0.35

    # 🔥 Filter schlechte Treffer raus
    ranked = [
    (doc, score) for doc, score in ranked
    if score > 0.3
    ]

    return ranked[:3]


def build_answer(query, passages):

    if not passages:
        return {
            "answer": "Keine Daten gefunden.",
            "sources": [],
            "confidence": 0.0
        }

    if passages[0][1] < 0.2:
        return {
            "answer": "Keine ausreichend relevanten Informationen gefunden.",
            "sources": [],
            "confidence": 0.0
        }

    confidence = float(passages[0][1])

    context = "\n\n".join(
        [f"[{i+1}] {doc[:300]}" for i, (doc, _) in enumerate(passages)]
    )

    prompt = f"""
Du bist ein präziser Assistent.

Antworte IMMER im JSON Format:

{{
  "status": "ok" | "missing_info",
  "answer": "...",
  "clarification_question": "..."
}}

REGELN:
1. Wenn wichtige Informationen fehlen → status = "missing_info"
   → dann KEINE Antwort, nur Rückfrage

2. Wenn ausreichend Kontext vorhanden ist → status = "ok"

3. Verwende ausschließlich Informationen aus dem Kontext.

4. Wenn im Kontext konkrete Zahlen, Mautwerte oder Regeln stehen,
   MUSST du diese exakt verwenden.

5. Erfinde niemals allgemeine Aussagen wie:
   - "keine Regelung vorhanden"
   - "variiert je nach Bundesland"
   - "nicht spezifiziert"

6. Wenn etwas im Kontext steht, aber unklar ist:
   → sage "nicht eindeutig im Dokument spezifiziert", aber erfinde nich

KONTEXT:
{context}

FRAGE:
{query}
"""

    response = llm_client.chat.completions.create(
        model="llama3.2:latest",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    import json
    raw = response.choices[0].message.content

    try:
        data = json.loads(raw)
    except:
        data = {
            "status": "ok",
            "answer": raw
        }

    # ✅ HIER RICHTIG EINGERÜCKT
    if "status" not in data:
        data["status"] = "ok"

    if "answer" not in data:
        data["answer"] = ""

    # ✅ LOGIK
    if data["status"] == "missing_info":
        final_answer = data.get("clarification_question", "Bitte präzisieren.")
    else:
        final_answer = data.get("answer", "")

    return {
        "answer": final_answer,
        "confidence": round(confidence, 3),
        "sources": [
            {"score": float(score), "text": doc[:300]}
            for doc, score in passages
        ]
    }

@app.post("/v1/chat/completions")
def chat(query: dict):

    user_id = get_user_id()
    user_message = query["messages"][-1]["content"]

    session = sessions.get(user_id, {})

    # 🧠 1. Prüfen ob Rückfrage offen ist
    if session.get("pending_question"):
        # Kontext kombinieren
        user_message = session["pending_question"] + " " + user_message
        session["pending_question"] = None

    passages = retrieve(user_message)
    result = build_answer(user_message, passages)

    # 🧠 2. Wenn Rückfrage nötig → speichern
    if "Was genau meinst du" in result["answer"]:
        session["pending_question"] = user_message

    sessions[user_id] = session

    return {
        "choices": [
            {
                "message": {
                    "content": result["answer"]
                }
            }
        ]
    }


@app.get("/v1/models")
def list_models():
    return {
        "data": [
            {
                "id": "rag-local",
                "object": "model",
                "owned_by": "local"
            }
        ]
    }
from openai import OpenAI

llm_client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

