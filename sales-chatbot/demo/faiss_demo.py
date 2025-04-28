import os

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

# 实例化文档加载器
loader = TextLoader("../text/sale_talks.txt",encoding='utf-8')
# 加载文档
documents = loader.load()

# 实例化文本分割器
text_splitter = CharacterTextSplitter(
    separator=r'\d+\.',
    chunk_size=100,
    chunk_overlap=0,
    length_function= len,
    is_separator_regex=True
)
# 分割文本
docs = text_splitter.split_documents(documents)

# OpenAI Embedding 模型
embeddings = OpenAIEmbeddings(
    api_key=os.getenv('YI_API_KEY'),
    base_url="https://vip.apiyi.com/v1"
)

# FAISS 向量数据库，使用 docs 的向量作为初始化存储
db = FAISS.from_documents(docs,embeddings)

# 构造提问
query = "贷款能办下来吗"
#
# # 1、在 Faiss 中进行相似度搜索，找出与 query 最相似结果
# result_docs = db.similarity_search(query)
# print(result_docs[0].page_content)

# 2、持久化存储 FaissDB
# db.save_local('faiss_index')

# 3、使用 参数 k 指定返回结果数量——定义一个检索器
# topK_retriever = db.as_retriever(search_kwargs = {"k":3})
# result_docs = topK_retriever.get_relevant_documents(query)
# for doc in result_docs:
#     print(doc.page_content+"\n")

# 4、使用 similarity_score_threshold 设置阈值，提升结果的相关性质量
# 实例化一个 similarity_score_threshold  Retriever
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs = {"score_threshold":0.65}
)

result_docs = retriever.get_relevant_documents("位置偏不偏")
for doc in result_docs:
    print(doc.page_content+'\n')


