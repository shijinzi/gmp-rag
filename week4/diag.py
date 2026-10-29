# -*- coding: utf-8 -*-
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from sentence_transformers import SentenceTransformer
import chromadb

embedder = SentenceTransformer("BAAI/bge-small-zh-v1.5")
client = chromadb.PersistentClient(path=r"D:\gmp-rag\week3\chroma_db")
col = client.get_collection(name="gmp_kb")

qv = embedder.encode(["设计和开发输入有哪些？"], normalize_embeddings=True).tolist()
res = col.query(query_embeddings=qv, n_results=10)
for i, (doc, meta, dist) in enumerate(zip(res["documents"][0], res["metadatas"][0], res["distances"][0])):
    flag = "  <== 含设计和开发输入" if "设计和开发输入" in doc else ""
    print(f"{i+1}. 第{meta['page']}页 相似度{1-dist:.3f}{flag}")
    print("   " + doc[:45].replace("\n", " "))