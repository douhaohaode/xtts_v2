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


def generate(clone_audio_path, text, language, temperature, length_penalty, repetition_penalty, top_k, top_p, num_gpt_outputs, gpt_cond_len, gpt_cond_chunk_len, max_ref_len, sound_norm_refs, gpt_batch_size, num_chars):

    config.temperature = temperature
    config.length_penalty = float(length_penalty)
    config.repetition_penalty = float(repetition_penalty)
    config.top_k = top_k
    config.top_p = top_p
    config.num_gpt_outputs = num_gpt_outputs
    config.gpt_cond_len = gpt_cond_len
    config.gpt_cond_chunk_len = gpt_cond_chunk_len

    config.max_ref_len = max_ref_len
    repair = False
    if len(sound_norm_refs) > 0:
        repair = True
    config.sound_norm_refs = repair

    config.model_args.gpt_batch_size = gpt_batch_size
    config.model_args.num_chars = num_chars
    print(config)

    outputs = model.synthesize(
        text,
        config,
        speaker_wav=clone_audio_path,
        language=language,
    )

    output_audio = ""
    output_audio = rename_tool.path("audio", "wav")
    torchaudio.save(output_audio, torch.tensor(outputs["wav"]).unsqueeze(0), 24000)
    return output_audio
