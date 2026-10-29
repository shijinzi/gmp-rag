# -*- coding: utf-8 -*-
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)

resp = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "你是医疗器械法规助手，回答简洁严谨。"},
        {"role": "user", "content": "用一句话说明什么是变更控制。"},
    ],
    temperature=0.3,
)
print(resp.choices[0].message.content)