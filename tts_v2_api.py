from TTS.api import TTS
import torch
import rename_tool

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
tts = tts.to(device)

# generate speech by cloning a voice using default settings
# def generate_api(clone_audio_path, text, language):
#     output_path = rename_tool.path("audio", "wav")
#     tts.tts_to_file(text=text,
#                     file_path=output_path,
#                     speaker_wav=clone_audio_path,
#                     language=language)
#     return output_path


# generate speech by cloning a voice using custom settings
def generate_api_custom(clone_audio_path, text, language, emotion, speed):
    output_path = rename_tool.path("audio", "wav")
    tts.tts_to_file(text=text,
                    file_path=output_path,
                    speaker_wav=clone_audio_path,
                    language=language,
                    emotion=emotion,
                    speed=speed,)
    return output_path
