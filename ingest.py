# -*- coding: utf-8 -*-
import os
import fitz
from rapidocr_onnxruntime import RapidOCR
from sentence_transformers import SentenceTransformer
import chromadb

# ================= 配置 =================
PDF_PATH  = r"D:\gmp-rag\week3\医疗器械质量管理体系用于法规的要求.pdf"   # ← 改成你的扫描件文件名
DB_DIR    = r"D:\gmp-rag\week3\chroma_db"          # 向量库存放位置
MODEL     = "BAAI/bge-small-zh-v1.5"         # 中文向量模型
# =========================================

# ---------- 第1步：OCR 识别扫描件 ----------
print("【1/4】OCR 识别中……")
ocr = RapidOCR()
pdf = fitz.open(PDF_PATH)
all_text = []
for page_num in range(pdf.page_count):
    page = pdf.load_page(page_num)
    pix = page.get_pixmap(dpi=300)
    result, _ = ocr(pix.tobytes("png"))
    all_text.append("\n".join(line[1] for line in result) if result else "")
pdf.close()
print(f"  OCR 完成，共 {len(all_text)} 页")

# ---------- 第2步：按页切块 ----------
print("【2/4】切分文档……")
# 每页固定为一块（或按需设阈值），带页码 metadata
chunks = []
for i, t in enumerate(all_text):
    page = i + 1
    if page >= 34:              # 34页起是附录表A.1等，对问答无价值，跳过
        continue
    if t.strip():
        chunks.append({"text": t, "page": page})
print(f"  切分得到 {len(chunks)} 个非空块")

# ---------- 第3步：向量化 ----------
print("【3/4】加载向量模型并计算向量(首次会下载模型，稍等）……")
model = SentenceTransformer(MODEL)
texts = [c["text"] for c in chunks]
vectors = model.encode(texts, normalize_embeddings=True).tolist()
print(f"  已完成 {len(vectors)} 个向量，每个维度 {len(vectors[0])}")

# ---------- 第4步：存入 Chroma ----------
print("【4/4】写入向量库……")
client = chromadb.PersistentClient(path=DB_DIR)          # 建/打开库
collection = client.get_or_create_collection(
    name="gmp_kb",
    metadata={"hnsw:space": "cosine"},                   # 用余弦相似度，适合语义检索
)
# 如果重复跑，先清空旧数据，避免叠加
try:
    collection.delete(where={"source": PDF_PATH})
except Exception:
    pass

collection.add(
    ids=[f"chunk_{i}" for i in range(len(chunks))],
    documents=texts,
    embeddings=vectors,
    metadatas=[{"page": c["page"], "source": PDF_PATH} for c in chunks],
)
print(f"  已写入 {collection.count()} 条到向量库 {DB_DIR}")
print("入库完成 ✅")