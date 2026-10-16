import json, os, faiss, numpy as np, openai

EMBED_MODEL = "gemini-embedding-001"
client = openai.OpenAI(api_key=os.environ["GEMINI_API_KEY"],
                       base_url="https://generativelanguage.googleapis.com/v1beta/openai/")

chunks = json.load(open("chunks.json", encoding="utf-8"))
r = client.embeddings.create(model=EMBED_MODEL, input=[f'{c["section"]}\n{c["text"]}' for c in chunks])
emb = np.array([e.embedding for e in r.data], dtype="float32")
emb /= np.linalg.norm(emb, axis=1, keepdims=True)

index = faiss.IndexFlatIP(emb.shape[1])
index.add(emb)
faiss.write_index(index, "index.faiss")
print(index.ntotal, "vectors, dim", emb.shape[1], "model", EMBED_MODEL)
