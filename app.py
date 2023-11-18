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

with gr.Blocks(css=css) as demo:
    gr.Markdown(f"### [coqui-ai-webui](https://github.com/douhaohaode/xtts)")
    with gr.Tab("文字转语音-本地"):  # TTS
        with gr.Column():
            input_text = gr.Textbox(label="输入文本", lines=4, placeholder="在此输入文字。")
            language = gr.Dropdown(language_list, label="语言", value=language_list[0])

        with gr.Column():
            with gr.Row():
                temperature = gr.Slider(0.0, 1.0, value=0.75, step=0.05,
                                        label="temperature 值越大越有創意 , 犧牲穩定性")
                length_penalty = gr.Slider(0, 2.0, value=1.0, step=0.1, label="length_penalty")
                repetition_penalty = gr.Slider(1.0, 100.0, value=10.0, step=1, label="repetition_penalty")

            with gr.Row():
                top_k = gr.Slider(1.0, 2000.0, value=50.0, step=1.0, label="top_k")
                top_p = gr.Slider(0.1, 0.99, value=0.85, step=0.05, label="top_p")
                num_gpt_outputs = gr.Slider(1.0, 50.0, value=1.0, step=1.0,
                                            label="num_gpt_outputs 值越大创建伟大事物的概率更高")

            with gr.Row():
                gpt_cond_len = gr.Slider(1.0, 600.0, value=30.0, step=1, label="gpt_cond_len")
                gpt_cond_chunk_len = gr.Slider(1.0, 600.0, value=4.0, step=1, label="gpt_cond_chunk_len")
                max_ref_len = gr.Slider(1.0, 60.0, value=10.0, step=1, label="max_ref_len")

            with gr.Row():
                sound_norm_refs = gr.CheckboxGroup(["调节"], label="是否规范调节音频")
                gpt_batch_size = gr.Slider(1.0, 10000.0, value=1.0, step=1.0, label="gpt_batch_size")
                num_chars = gr.Slider(1.0, 1024.0, value=255.0, step=1, label="num_chars")

        with gr.Row():
            audio_filename = gr.Audio(label="Input audio.wav", type='filepath')
            output_audio = gr.Audio(label="生成的音频1", type='filepath')

        with gr.Row():
            clone_voice_button = gr.Button("创建音频文件")
            clone_voice_button.click(tts_v2.generate,
                                     inputs=[audio_filename, input_text, language, temperature, length_penalty,
                                             repetition_penalty, top_k, top_p,
                                             num_gpt_outputs, gpt_cond_len, gpt_cond_chunk_len, max_ref_len,
                                             sound_norm_refs, gpt_batch_size, num_chars],
                                     outputs=output_audio)
    with gr.Tab("文字转语音-API"):  # TTS
        with gr.Column():
            input_text_api = gr.Textbox(label="输入文本", lines=4, placeholder="在此输入文字")  # Input Text
            language_api = gr.Dropdown(language_list, label="语言", value=language_list[0], )
        with gr.Row():
            emotion = gr.Radio(emotion_list, label="emotion", value=emotion_list[0], info="模型的情感")
            speed = gr.Slider(0.0, 2.0, value=0, step=0.1, label="speed", info="速度系数")

        with gr.Row():
            audio_filename_api = gr.Audio(label="Input audio.wav", type='filepath')
            output_audio_api = gr.Audio(label="生成的音频", type="filepath")
        with gr.Row():
            clone_voice_button_api = gr.Button("创建音频文件")
            clone_voice_button_api.click(tts_v2_api.generate_api_custom,
                                         inputs=[audio_filename_api, input_text_api, language_api, emotion, speed],
                                         outputs=output_audio_api)

demo.launch()
