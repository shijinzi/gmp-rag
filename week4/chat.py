# -*- coding: utf-8 -*-
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import chromadb

# ===== 配置 =====
MODEL   = "BAAI/bge-small-zh-v1.5"
DB_DIR  = r"D:\gmp-rag\week3\chroma_db"      # 直接用第三周的库
K       = 5
THRESH  = 0.5
LLM     = "deepseek-chat"

embedder = SentenceTransformer(MODEL)
client = chromadb.PersistentClient(path=DB_DIR)
collection = client.get_collection(name="gmp_kb")
llm = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")

# 检索：返回 [(页码, 文本, 相似度)]，已做阈值过滤
def retrieve(query, k=K):
    qv = embedder.encode([query], normalize_embeddings=True).tolist()
    res = collection.query(query_embeddings=qv, n_results=k)
    out = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        sim = 1 - dist
        if sim >= THRESH:
            out.append((meta["page"], doc, sim))
    return out

# 把检索块拼成带编号的参考资料
def build_context(hits):
    blocks = []
    for i, (page, text, sim) in enumerate(hits):
        blocks.append(f"[资料{i+1} 第{page}页]\n{text}")
    return "\n\n".join(blocks)

SYSTEM_PROMPT = """你是医疗器械质量管理体系(GB/T 42061 / ISO 13485)法规咨询助手。
请严格遵守：
1. 只能依据下面提供的"参考资料"回答，不得使用资料外的知识、不得编造条款。
2. 回答中引用的内容,要在句末用(资料N,第X页)标注来源。
3. 资料不足以回答时，明确说"根据现有资料无法回答该问题"，不要猜测。
4. 回答条理清晰，必要时分点，使用中文。"""

def answer(query, hits):
    context = build_context(hits)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"参考资料：\n{context}\n\n问题:{query}"},
    ]
    resp = llm.chat.completions.create(model=LLM, messages=messages, temperature=0.3)
    return resp.choices[0].message.content
def answer_guard(query):
    hits = retrieve(query)
    if not hits:
        return None, "根据现有资料无法回答该问题（知识库中未检索到相关条款）。"
    return hits, answer(query, hits)
def chat_loop():
    history = []
    print("GMP 法规问答已启动。输入问题开始；命令：/clear 清空历史，/exit 退出")
    while True:
        try:
            q = input("\n你:").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q == "/exit":
            break
        if q == "/clear":
            history = []
            print("（对话历史已清空）")
            continue

        hits = retrieve(q)
        if not hits:
            print("助手：根据现有资料无法回答该问题（知识库中未检索到相关条款）。")
            continue

        context = build_context(hits)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"参考资料：\n{context}\n\n问题:{q}"},
        ]
        # 把最近6轮历史插到当前问题前
        messages = [messages[0]] + history[-12:] + [messages[1]]
        resp = llm.chat.completions.create(model=LLM, messages=messages, temperature=0.3)
        ans = resp.choices[0].message.content
        print(f"助手：{ans}")
        history.append({"role": "user", "content": q})
        history.append({"role": "assistant", "content": ans})

if __name__ == "__main__":
    chat_loop()
