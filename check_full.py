# -*- coding: utf-8 -*-
import fitz
from rapidocr_onnxruntime import RapidOCR
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

PDF_PATH = r"D:\gmp-rag\week3\医疗器械质量管理体系用于法规的要求.pdf"   # 换成你的扫描件名

# 1. OCR（和之前一样）
ocr = RapidOCR()
pdf = fitz.open(PDF_PATH)
all_text = []
for page_num in range(pdf.page_count):
    page = pdf.load_page(page_num)
    pix = page.get_pixmap(dpi=300)
    result, _ = ocr(pix.tobytes("png"))
    all_text.append("\n".join(line[1] for line in result) if result else "")
pdf.close()

docs = [Document(page_content=t, metadata={"page": i+1}) for i, t in enumerate(all_text)]

# 2. 切分
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=100,
    separators=["\n\n", "\n", "。", "；", " ", ""],
)
chunks = splitter.split_documents(docs)
print(f"共 {len(chunks)} 块\n")

# 3. 完整打印前 5 块（不截断），标字符数，方便检查完整性
for i, c in enumerate(chunks[:5]):
    print(f"===== 块 {i+1} | 页码 {c.metadata['page']} | 长度 {len(c.page_content)} 字符 =====")
    print(c.page_content)          # 完整内容
    print()