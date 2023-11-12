import rename_tool
import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

config = XttsConfig()
config.load_json("./source/model_v2/config.json")
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir="./source/model_V2/", eval=True)
model.cuda()


def generate(clone_audio_path, text, language):
    outputs = model.synthesize(
        text,
        config,
        speaker_wav=clone_audio_path,
        # gpt_cond_len=gpt_cond_len,
        language=language,
        # temperature=temperature,
        # length_penalty=float(length_penalty),
        # repetition_penalty=float(length_penalty),
        # top_k=top_k,
    )

    output_audio = rename_tool.path("audio", "wav")
    torchaudio.save(output_audio, torch.tensor(outputs["wav"]).unsqueeze(0), 24000)
    return output_audio
