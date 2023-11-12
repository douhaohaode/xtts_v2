import gradio as gr
import tts_v2
import tts_v2_api
css = """
#warning {background-color: #FFCCCB}
.gradio-container {background-color: black}
.feedback textarea {font-size: 24px !important}
"""

language_list = ['zh-cn', 'en', 'fr', 'de', 'it', 'pt', 'pl', 'tr', 'ru', 'nl', 'cs', 'ar', 'es', 'hu', 'ko', 'ja']

emotion_list = ["Neutral", "Happy", "Sad", "Angry", "Dull"]

def clone_voice(clone_audio_path, text, language):
    path = tts_v2.generate(clone_audio_path, text, language)
    return path


with gr.Blocks(css=css) as demo:
    gr.Markdown(f"### [coqui-ai-webui](https://github.com/douhaohaode/xtts)")
    with gr.Tab("文字转语音-本地"):  # TTS
        with gr.Column():
            input_text = gr.Textbox(label="输入文本", lines=4, placeholder="在此输入文字。")
            language = gr.Dropdown(language_list, label="语言", value=language_list[0])
        with gr.Row():
            audio_filename = gr.Audio(label="Input audio.wav", type='filepath')
            output_audio = gr.Audio(label="生成的音频", type="filepath")

            # with gr.Column():
            #     language = gr.Dropdown(language_list, label="语言", value=language_list[0])
            #     temperature = gr.Slider(0.0, 1.0, value=0.65, step=0.05, label="temperature", info="自回归模型的softmax温度")
            #     length_penalty = gr.Slider(1.0, 10.0, value=1.0, step=0.1, label="length_penalty",
            #                                info="应用于自回归解码器的长度惩罚。 较高的设置会导致模型产生更简洁的输出。")
            #     repetition_penalty = gr.Slider(1.0, 10.0, value=2.0, step=0.1, label="repetition_penalty" info="自回归"
            #                                      "解码器在执行期间重复自身的惩罚解码。可用于减少长时间沉默或“us”等的发生率。")
            #     top_k = gr.Slider(1.0, 99999.0, value=50.0, step=2.0, label="top_k",  info="采样中使用的 K 值。 [0，无限]。"
            #                                     " 较低的值意味着解码器产生更多“可能”的结果" "（又名无聊）输出。 默认为 50。")
            #     gpt_cond_len = gr.Slider(1.0, 100.0, value=6.0, step=0.1, label="gpt_cond_len",  info="用于克隆的音频长度)
        with gr.Row():
            clone_voice_button = gr.Button("创建音频文件")
            clone_voice_button.click(clone_voice, inputs=[audio_filename, input_text, language],
                                     outputs=output_audio)
    with gr.Tab("文字转语音-API"):  # TTS
        with gr.Column():
            input_text_api = gr.Textbox(label="输入文本", lines=4, placeholder="在此输入文字")  # Input Text
            language_api = gr.Dropdown(language_list, label="语言", value=language_list[0], )
        with gr.Row():
            emotion = gr.Radio(emotion_list, label="emotion", value=emotion_list[0], info="模型的情感")
            speed = gr.Slider(0.0, 2.0, value=0, step=0.2, label="speed", info="速度系数")

        with gr.Row():
            audio_filename_api = gr.Audio(label="Input audio.wav", type='filepath')
            output_audio_api = gr.Audio(label="生成的音频", type="filepath")
        with gr.Row():
            clone_voice_button_api = gr.Button("创建音频文件")
            clone_voice_button_api.click(tts_v2_api.generate_api_custom, inputs=[audio_filename_api, input_text_api, language_api, emotion, speed],
                                         outputs=output_audio_api)

demo.launch()
