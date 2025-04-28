# Faiss 向量数据库初始化
import os

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

loader = TextLoader('../text/sale_talks.txt',encoding='utf-8')
docs = loader.load()

text_splitter = CharacterTextSplitter(
    separator=r'\d+\.',
    chunk_size=100,
    chunk_overlap=0,
    length_function=len,
    is_separator_regex=True
)

splitter_docs=text_splitter.split_documents(docs)

embedding= OpenAIEmbeddings(
    api_key=os.getenv('YI_API_KEY'),
    base_url="https://vip.apiyi.com/v1"
)

# 根据 docs创建向量数据库
db = FAISS.from_documents(splitter_docs,embedding)

# 将向量数据库保存到本地
db.save_local('sale_faiss')