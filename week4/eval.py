# -*- coding: utf-8 -*-
from chat import answer_guard

QUESTIONS = [
    "变更控制流程是什么？", "设计和开发输入有哪些要求？",
    "设计和开发输出包括什么？", "管理评审的输入有哪些？",
    "内部审核多久进行一次？", "采购产品验证怎么做？",
    "不合格品如何控制？", "纠正措施的流程是什么？",
    "文件控制有哪些要求？", "无菌器械有什么专用要求？",
]
hit = 0
for q in QUESTIONS:
    hits, ans = answer_guard(q)
    ok = bool(hits) and "无法回答" not in ans
    hit += ok
    print(f"[{'OK' if ok else 'X'}] {q}")
    print(f"    {ans[:80]}...\n")
print(f"命中率：{hit}/{len(QUESTIONS)} = {hit/len(QUESTIONS):.0%}")
