from VectorBase import VectorStore
from utils import ReadFiles
from LLM import OpenAIChat
from Embeddings import OpenAIEmbedding
import os

# 确保data目录存在
if not os.path.exists('./data'):
    os.makedirs('./data')
    print("请注意：已创建空的data目录，请将文档文件放入其中")

# 读取文档
docs = ReadFiles('./data').get_content(max_token_len=600, cover_content=150) # 获得data目录下的所有文件内容并分割

if not docs:
    print("警告：data目录中没有找到支持的文档文件（支持格式：.md, .txt, .pdf）")
    print("请在data目录中添加文档文件后重新运行")
    exit()

vector = VectorStore(docs)
embedding = OpenAIEmbedding() # 创建EmbeddingModel
vector.get_vector(EmbeddingModel=embedding)
vector.persist(path='storage') # 将向量和文档内容保存到storage目录下，下次再用就可以直接加载本地的数据库

# vector.load_vector('./storage') # 加载本地的数据库

question = 'RAG的原理是什么？'

results = vector.query(question, EmbeddingModel=embedding, k=1)
if not results:
    print("未找到相关文档内容")
    exit()
content = results[0]
chat = OpenAIChat(model='Qwen/Qwen2.5-32B-Instruct')
print(chat.chat(question, [], content))