import os

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    api_key = os.getenv('YI_API_KEY'),
    base_url = "https://vip.apiyi.com/v1"
)
# 加载 FaissDB
new_db = FAISS.load_local('faiss_index',embeddings,allow_dangerous_deserialization=True)

query='贷款能办下来吗'
result_docs = new_db.similarity_search(query)
print(result_docs[0].page_content)