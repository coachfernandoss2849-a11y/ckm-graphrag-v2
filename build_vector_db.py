# build_vector_db.py
import os
import shutil
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# 【引入极其稳定的云端模型，替换原有的 HuggingFace】
from langchain_community.embeddings import ZhipuAIEmbeddings

print("🚀 开始构建本地医学向量数据库...")

# 填入你刚刚申请的智谱 API Key
os.environ["ZHIPUAI_API_KEY"] = "1cddba76ebff472d97774d5b55fabd3c.s1DzJye0WJKfbaDg" # <--- 填在这里！

# 如果已经有旧的 chroma_db，先删掉，防止两种不同格式的向量混在一起打架
db_path = "./chroma_db"
if os.path.exists(db_path):
    print("🗑️ 检测到旧数据库，正在清理...")
    shutil.rmtree(db_path)

# 加载你的 6 篇精编英文文献
loader = DirectoryLoader('./knowledge_base', glob="**/*.txt", loader_cls=TextLoader)
documents = loader.load()
print(f"📖 共读取到 {len(documents)} 篇医学文献。")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

# 使用智谱的在线 Embedding 引擎，速度极快，不吃电脑配置，永不断联！
print("⚡ 正在连接国内智谱云端引擎...")
embeddings_model = ZhipuAIEmbeddings(model="embedding-3")

print("💾 正在存入新数据库，请稍候...")
db = Chroma.from_documents(documents=chunks, embedding=embeddings_model, persist_directory=db_path)

print("🎉 恭喜！全新的极速版知识库构建完成！")
