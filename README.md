**简单实现RAG项目**

  utils.py - 读取data文件夹下的文档并对其中文本分块处理，得到分块后的文本块列表

  Embeddings.py - 根据调用的API的LLM中的embedding模型将输入的文档text向量化

  VertorBase.py - 创建向量数据库，存放文档片段和对应的向量表示，并设计一个检索模块用于根据Query检索相关文档片段

  LLM.py - 创建完整的RAG（检索增强生成）智能问答系统模型类

  demo.py - 简单问题的RAG检索问答

  demo_interactive.py - 交互式的RAG检索问答

  data - 存储RAG相关文档
