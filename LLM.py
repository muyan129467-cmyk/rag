#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from typing import Dict, List, Optional, Tuple, Union
from openai import OpenAI

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

RAG_PROMPT_TEMPLATE="""
使用以上下文来回答用户的问题。如果你不知道答案，就说你不知道。总是使用中文回答。
问题: {question}
可参考的上下文：
···
{context}
···
如果给定的上下文无法让你做出回答，请回答数据库中没有这个内容，你不知道。
有用的回答:
"""


class BaseModel:
    def __init__(self, model) -> None:
        self.model = model

    def chat(self, prompt: str, history: List[dict], content: str) -> str:
        pass

    def load_model(self):
        pass

class OpenAIChat(BaseModel):
    def __init__(self, model: str = "Qwen/Qwen2.5-32B-Instruct") -> None:
        self.model = model
        self.client = OpenAI()
        self.client.api_key = os.getenv("OPENAI_API_KEY")   
        self.client.base_url = os.getenv("OPENAI_BASE_URL")

    def chat(self, question: str, history: List[dict], content: str) -> str:
        """修改参数名从 prompt 改为 question，以匹配 ConversationManager 的调用"""
        # 构建包含上下文的用户消息
        user_message = RAG_PROMPT_TEMPLATE.format(question=question, context=content)
        
        # 复制历史记录并添加新消息
        messages = history.copy()
        messages.append({'role': 'user', 'content': user_message})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=2048,
                temperature=0.1
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"API调用失败: {e}")
            return "抱歉，生成回答时出现错误。"