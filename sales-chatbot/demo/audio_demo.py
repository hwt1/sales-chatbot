import gradio as gr
import openai
import os

from openai import OpenAI

# 设置 OpenAI API 密钥
# openai.api_key = "your-api-key-here"

client = OpenAI(
    base_url="https://vip.apiyi.com/v1",
    api_key = os.getenv('YI_API_KEY')
)
# 实时录音并转文字的函数
def transcribe_audio(audio):
    """将音频转录为文本"""
    audio_file = open(audio, "rb")
    response = client.audio.translations.create(model = "whisper-1", file=audio_file,prompt='保留原有的语言')
    print(response)
    return response.text


# 文本到语音的函数
def text_to_speech(text):
    """将文本转换为语音并保存"""
    with client.audio.speech.with_streaming_response.create(model='tts-1',voice="echo",input=text) as res:
        res.stream_to_file("output.mp3")
    return "output.mp3"


# 创建 Gradio 接口
def gradio_interface(audio, text):
    # 先转录音频为文本
    transcribed_text = transcribe_audio(audio)

    # 生成语音并播放
    audio_output = text_to_speech(text or transcribed_text)

    return transcribed_text, audio_output


# Gradio 界面定义
iface = gr.Interface(
    fn=gradio_interface,
    inputs=[gr.Audio(sources=["microphone"], type='filepath'), gr.Textbox()],
    outputs=[gr.Textbox(), gr.Audio( type='filepath')],
    live=True
)

# 启动 Gradio 应用
iface.launch()
