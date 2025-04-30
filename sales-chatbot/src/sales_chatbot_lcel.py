
# 增加 LCEL 的写法
import gradio as gr

from langchain import hub
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tracers import ConsoleCallbackHandler
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from config import OPENAI_API_KEY, OPENAI_BASE_URL


def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

def init_sales_bot_lcel(vector_store_dir:str = 'sale_faiss'):
    embedding = OpenAIEmbeddings(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    db = FAISS.load_local(vector_store_dir,embedding,allow_dangerous_deserialization=True)
    retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs = {"score_threshold":0.7}
    )


    llm = ChatOpenAI(
        model='gpt-4o-mini',
        api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL
    )
    prompt = hub.pull("rlm/rag-prompt")

    # 构造chain
    rag_chain = (
        {'context':retriever|format_docs ,'question':RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    # print('==================')
    # print(retriever.input_schema.model_json_schema())
    # print('==================')
    # print(rag_chain.input_schema.model_json_schema())

    global SALES_BOT
    SALES_BOT = rag_chain



def sales_chat_stream(history):
    print(f"[history]{history}")
    message = history[-1]['content']
    print(f"[message]{message}")


    # !!!!!!!! 创建回调处理器，用于输出 chain 内部的调用流程
    callback_handler = ConsoleCallbackHandler()
    history.append({"role": "assistant", "content": ""})
    for chunk in SALES_BOT.stream(message,config={"callbacks": [callback_handler]}):
        history[-1]['content'] += chunk
        yield history # 逐步返回，让前端实时显示

# 5、创建 gradio界面
def launch_gradio():
    with gr.Blocks() as app:
        chatbot = gr.Chatbot(type="messages")
        msg = gr.Textbox(label='输入你的问题')

        def user_submit(user_message, history: list):
            return "", history + [{"role": "user", "content": user_message}]

        msg.submit(user_submit,[msg,chatbot],[msg,chatbot],queue=False
                   ).then(sales_chat_stream,chatbot,chatbot)
    app.launch(share=True,server_name='127.0.0.1')


if __name__ == '__main__':
    # 初始化房产销售机器人
    init_sales_bot_lcel()
    # 启动 gradio服务
    launch_gradio()