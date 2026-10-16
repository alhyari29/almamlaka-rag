import json, faiss, numpy as np
from sentence_transformers import SentenceTransformer

chunks = json.load(open("chunks.json", encoding="utf-8"))
model = SentenceTransformer("BAAI/bge-m3")
emb = model.encode([f'{c["section"]}\n{c["text"]}' for c in chunks],
                   normalize_embeddings=True, batch_size=8, show_progress_bar=True)
index = faiss.IndexFlatIP(emb.shape[1])
index.add(np.asarray(emb, dtype="float32"))
faiss.write_index(index, "index.faiss")
print(index.ntotal, "vectors, dim", emb.shape[1])

for q in ["What is the total budget?", "ما هي الميزانية الإجمالية؟",
          "What is the weather in Amman?"]:
    qv = model.encode([q], normalize_embeddings=True).astype("float32")
    s, i = index.search(qv, 3)
    print("\nQ:", q)
    for score, idx in zip(s[0], i[0]):
        print(f"  {score:.3f}  {chunks[idx]['id']}")
