# 1、加载出向量数据库
import os

import gradio as gr
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.tracers import ConsoleCallbackHandler
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from openai import OpenAI

from config import OPENAI_API_KEY, OPENAI_BASE_URL

def init_sales_bot(vector_store_dir:str ='sale_faiss'):
    embedding = OpenAIEmbeddings(
        api_key = OPENAI_API_KEY,
        base_url = OPENAI_BASE_URL
    )
    db = FAISS.load_local(vector_store_dir,embedding,allow_dangerous_deserialization=True)
    # 2、创建带有相似度阈值的检索器
    retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs = {"score_threshold":0.7})
    # 3、创建LLM
    llm = ChatOpenAI(
        model_name='gpt-4o',
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        temperature=0.5,
        max_tokens=1000,
        verbose=True
    )

    # 1. system prompt：只定义身份，不嵌套 context
    system_prompt = """
    你是一位专业的房地产销售顾问，这是你的固定身份设定。
    只有在用户明确询问你的身份、职业、角色、背景时，才自我介绍为：“我是本项目的资深房产顾问，很高兴为您服务。”

    请根据提供的参考资料（Context）回答客户问题。
    - 当Context中存在相关内容时，尽量准确引用。
    - 当Context中没有完全匹配的问题，但相关信息可以支持合理归纳总结时，允许基于已有内容进行简要总结，不得捏造或虚构。
    - 当Context确实无关时，礼貌地表示不知道。

    回答时不要主动提及你的身份，也不要暴露自己是AI助手。
    每个回答最多使用三句话，保持回答简洁、自然、有条理。
    
    Context:
    {context}
    """

    # 2. prompt结构
    prompt = ChatPromptTemplate.from_messages(
        [
           SystemMessagePromptTemplate.from_template(template=system_prompt),
           HumanMessagePromptTemplate.from_template(template="{input}")
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    retriever_chain = create_retrieval_chain(retriever, question_answer_chain)


    #
    # # 4、创建 chain
    # retriever_chain = RetrievalQA.from_chain_type(llm,retriever=retriever)

    # 输出内部 Chain 的日志
    #retriever_chain.combine_docs_chain.verbose = True
    # 返回向量数据库的检索结果
    #retriever_chain.return_source_documents = True

    global SALES_BOT
    SALES_BOT = retriever_chain


def init_openai():
    openai_client = OpenAI(
        api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL
    )

    global OPENAI_CLIENT
    OPENAI_CLIENT = openai_client

# 实时录音并转文字的函数
def transcribe_audio(audio):
    """将音频转录为文本"""
    audio_file = open(audio, "rb")
    response = OPENAI_CLIENT.audio.transcriptions.create(model = "whisper-1", file=audio_file)
    print(response)
    return response.text

# 文本到语音的函数
def text_to_speech(text):
    """将文本转换为语音并保存"""
    with OPENAI_CLIENT.audio.speech.with_streaming_response.create(model='tts-1',voice="alloy",input=text) as res:
        res.stream_to_file("output.mp3")
    return "output.mp3"

def audio_chat(audio):
    if audio is None:
        return '',None
    # 1、从语音中获取文本
    query = transcribe_audio(audio)
    # 2、根据查询问题获取结果
    callback_handler = ConsoleCallbackHandler()
    ans = SALES_BOT.invoke({'input': query}, config={"callbacks": [callback_handler]})
    answer_text = ans['answer']
    # 3、将咨询回答转为语音
    ans_audio_path = text_to_speech(answer_text)

    return answer_text,ans_audio_path

def sales_chat(message,history):
    print(f"[message]{message}")
    print(f"[history]{history}")

    #ans = SALES_BOT({'query':message})
    # 创建回调处理器，用于输出 chain内部的调用流程
    callback_handler = ConsoleCallbackHandler()
    ans = SALES_BOT.invoke({'input':message},config={"callbacks": [callback_handler]})

    enable_chat = True
    # 如果检索出结果，或者开了大模型聊天模式
    if ans['context'] or enable_chat:
        print(f"[result]{ans['answer']}")
        print(f"[source_documents]{ans['context']}")
        return ans['answer']
        # 否则输出套路话术
    else:
        return "这个问题我要问问领导"
# 5、创建 gradio界面
def launch_gradio():
    chat_tab = gr.ChatInterface(fn=sales_chat,title='房产销售')

    simple_audio_tab=gr.Interface(
        fn=audio_chat,
        inputs=[gr.Audio(sources=["microphone"], type='filepath')],
        outputs=[gr.Textbox(), gr.Audio(type='filepath')],
        live=True
    )
    app = gr.TabbedInterface([chat_tab, simple_audio_tab], ["文字对话", "简易语音对话"])
    app.launch(share=True,server_name='127.0.0.1')

if __name__ == '__main__':
    # 初始化房产销售机器人
    init_sales_bot()
    init_openai()
    # 启动 gradio服务
    launch_gradio()
