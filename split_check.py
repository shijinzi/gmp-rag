# -*- coding: utf-8 -*-
import fitz  # PyMuPDF：把 PDF 页转成图片
from rapidocr_onnxruntime import RapidOCR
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

PDF_PATH = r"D:\gmp-rag\week3\医疗器械质量管理体系用于法规的要求.pdf"   # ← 改成你的扫描件文件名

# 1. 打开 PDF，逐页转成图片并 OCR
ocr = RapidOCR()
all_text = []

pdf = fitz.open(PDF_PATH)
for page_num in range(pdf.page_count):
    page = pdf.load_page(page_num)
    # 把当前页渲染成高分辨率图片（清晰度越高识别越准）
    pix = page.get_pixmap(dpi=300)
    img_bytes = pix.tobytes("png")

    # 识别这一页的文字
    result, _ = ocr(img_bytes)
    if result:   # result 是每行识别框的列表
        page_text = "\n".join([line[1] for line in result])
    else:
        page_text = ""
    all_text.append(page_text)
    print(f"第 {page_num+1}/{pdf.page_count} 页识别完成")
pdf.close()

# 2. 把每页文字包成 Document（带页码 metadata）
docs = [Document(page_content=t, metadata={"page": i+1, "source": PDF_PATH})
        for i, t in enumerate(all_text)]
print(f"OCR 完成，共 {len(docs)} 页")

# 3. 预览第一页识别效果
print("第一页预览：")
print(docs[0].page_content[:200])
print("=" * 60)

# 4. 切分（和之前一样）
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n", "\n", "。", "；", " ", ""],
)
chunks = splitter.split_documents(docs)
print(f"切分得到 {len(chunks)} 块\n")

# 5. 打印前 10 块
for i, c in enumerate(chunks[:10]):
    print(f"【块 {i+1}】页码: {c.metadata.get('page')}")
    print(c.page_content[:120].replace("\n", " "))
    print("-" * 60)