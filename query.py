# -*- coding: utf-8 -*-
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"   # 保持镜像，模型已在本地缓存，基本不再联网
from sentence_transformers import SentenceTransformer
import chromadb

MODEL  = "BAAI/bge-small-zh-v1.5"
DB_DIR = r"D:\gmp-rag\week3\chroma_db"

# 加载同一个模型（现在从本地缓存读，很快）
model = SentenceTransformer(MODEL)

# 打开向量库
client = chromadb.PersistentClient(path=DB_DIR)
collection = client.get_collection(name="gmp_kb")
print(f"向量库共 {collection.count()} 条\n")

# 检索函数：输入问题 → 取回最相关的 3 块
def search(query, k=5, threshold=0.55):
    q_vec = model.encode([query], normalize_embeddings=True).tolist()
    res = collection.query(query_embeddings=q_vec, n_results=k)
    print(f"问题：{query}")
    print("-" * 50)
    for i, (doc, meta) in enumerate(zip(res["documents"][0], res["metadatas"][0])):
        sim = 1 - res["distances"][0][i]
        if sim < threshold:      # 相似度低于阈值的，不要
            print(f"  (过滤：相似度 {sim:.3f} 过低，丢弃)")
            continue
        print(f"[{i+1}] 页码 {meta['page']} | 相似度 {sim:.3f}")
        print(doc[:120].replace("\n", " "))
        print()
    print("=" * 60)

# 测试 3 个不同的问题
search("变更控制流程是什么？")
search("设计和开发输入的要求是什么？")
search("无菌医疗器械的专用要求")