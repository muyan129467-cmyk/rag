#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
from utils import ReadFiles
from Embeddings import OpenAIEmbedding
from LLM import OpenAIChat
from VectorBase import VectorStore


class ConversationManager:
    """管理多轮对话的历史记录"""
    
    def __init__(self, vector_store, embedder, chat_model):
        self.vector_store = vector_store  # 向量数据库
        self.embedder = embedder  # 嵌入模型
        self.chat_model = chat_model  # 聊天模型
        self.history = []  # 存储对话历史
    
    def ask(self, question, k=3):
        """提问并获取回答"""
        # 1. 检索相关文档
        relevant_docs = self.vector_store.query(question, self.embedder, k=k)
        context = "\n".join(relevant_docs)
        
        # 2. 生成回答
        answer = self.chat_model.chat(
            question=question,
            history=self.history.copy(),
            content=context
        )
        
        # 3. 更新历史记录
        self.history.append({'role': 'user', 'content': question})
        self.history.append({'role': 'assistant', 'content': answer})
        
        # 4. 限制历史长度
        self._trim_history()
        
        return answer, relevant_docs
    
    def _trim_history(self, max_messages=20):
        """修剪历史，保留最近的消息"""
        if len(self.history) > max_messages:
            self.history = self.history[-max_messages:]
    
    def clear_history(self):
        """清空对话历史"""
        self.history = []
    
    def get_history_display(self):
        """获取格式化的对话历史"""
        display = []
        for msg in self.history:
            role = "用户" if msg['role'] == 'user' else "助手"
            display.append(f"{role}: {msg['content']}")
        return "\n\n".join(display)
    
    def get_recent_history(self, count=3):
        """获取最近的几条历史记录"""
        recent = self.history[-min(count*2, len(self.history)):]  # 每轮对话有user和assistant两条
        return recent


def main():
    """主函数：交互式对话系统"""
    
    # 1. 初始化组件
    print("=" * 50)
    print("RAG 多轮对话系统")
    print("=" * 50)
    
    # 检查数据目录
    if not os.path.exists('./data'):
        os.makedirs('./data')
        print("⚠️  注意：已创建空的data目录，请将文档文件放入其中")
        print("支持的格式：.md, .txt, .pdf")
        return
    
    # 读取文档
    print("正在读取文档...")
    docs = ReadFiles('./data').get_content(max_token_len=600, cover_content=150)
    
    if not docs:
        print("❌ 错误：data目录中没有找到支持的文档文件")
        print("请在data目录中添加文档文件后重新运行")
        return
    
    print(f"✓ 已读取 {len(docs)} 个文档块")
    
    # 检查是否已有向量存储
    if os.path.exists('./storage/document.json'):
        print("检测到已有的向量存储，正在加载...")
        vector = VectorStore([])
        vector.load_vector(path='storage')
        print("✓ 向量存储加载完成")
    else:
        print("正在创建向量存储...")
        vector = VectorStore(docs)
        embedding = OpenAIEmbedding()
        vector.get_vector(EmbeddingModel=embedding)
        vector.persist(path='storage')
        print("✓ 向量存储创建完成")
    
    # 初始化模型
    print("正在初始化模型...")
    embedder = OpenAIEmbedding()
    chat_model = OpenAIChat(model='Qwen/Qwen2.5-32B-Instruct')
    
    # 创建对话管理器
    conversation = ConversationManager(vector, embedder, chat_model)
    
    print("\n" + "=" * 50)
    print("系统初始化完成！开始对话吧！")
    print("输入 'quit' 或 'exit' 退出程序")
    print("输入 'clear' 清空对话历史")
    print("输入 'history' 查看对话历史")
    print("输入 'help' 查看帮助")
    print("=" * 50 + "\n")
    
    # 2. 交互式对话循环
    while True:
        try:
            # 获取用户输入
            user_input = input("\n👤 你: ").strip()
            
            if not user_input:
                continue
            
            # 处理特殊命令
            if user_input.lower() in ['quit', 'exit', '退出']:
                print("👋 再见！")
                break
            
            elif user_input.lower() == 'clear':
                conversation.clear_history()
                print("✓ 对话历史已清空")
                continue
            
            elif user_input.lower() == 'history':
                history_text = conversation.get_history_display()
                if history_text:
                    print("\n📜 对话历史:")
                    print("-" * 40)
                    print(history_text)
                    print("-" * 40)
                else:
                    print("📭 对话历史为空")
                continue
            
            elif user_input.lower() == 'help':
                print("\n📋 可用命令:")
                print("  'quit', 'exit', '退出' - 退出程序")
                print("  'clear' - 清空对话历史")
                print("  'history' - 查看完整对话历史")
                print("  'help' - 显示此帮助信息")
                print("\n💡 提示: 直接输入问题即可开始对话")
                continue
            
            # 处理普通问题
            print("\n🤖 正在思考...", end='', flush=True)
            
            # 获取回答
            answer, relevant_docs = conversation.ask(user_input, k=3)
            
            print(f"\r{' ' * 30}\r", end='')  # 清空"正在思考..."提示
            
            # 显示回答
            print(f"🤖 助手: {answer}")
            
            # 可选：显示检索到的文档（用于调试）
            debug_mode = False  # 设置为True可查看检索结果
            if debug_mode and relevant_docs:
                print("\n📄 检索到的相关文档:")
                print("-" * 40)
                for i, doc in enumerate(relevant_docs[:2]):  # 只显示前2个
                    preview = doc[:200] + "..." if len(doc) > 200 else doc
                    print(f"文档 {i+1}: {preview}")
                print("-" * 40)
        
        except KeyboardInterrupt:
            print("\n\n👋 已中断，再见！")
            break
        
        except Exception as e:
            print(f"\n❌ 发生错误: {e}")
            print("请检查网络连接和API配置")


if __name__ == "__main__":
    main()