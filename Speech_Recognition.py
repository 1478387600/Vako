from transformers import pipeline
from transformers.pipelines.audio_utils import ffmpeg_microphone_live
from display import *

asr_model_id = "openai/whisper-tiny.en"
transcriber = pipeline("automatic-speech-recognition",
                       model=asr_model_id,
                       device="cpu")


def transcribe_mic(chunk_length_s: float) -> str:
    """ Transcribe the audio from a microphone """
    global transcriber
    sampling_rate = transcriber.feature_extractor.sampling_rate
    mic = ffmpeg_microphone_live(
        sampling_rate=sampling_rate,
        chunk_length_s=chunk_length_s,
        stream_chunk_s=chunk_length_s,
    )

    result = ""
    for item in transcriber(mic):
        result = item["text"]
        if not item["partial"][0]:
            break
    return result.strip()


def asr_init():
    # 直接在这里初始化 ASR，不需要额外的 asr_init()
    asr_model_id = "openai/whisper-tiny.en"
    transcriber = pipeline("automatic-speech-recognition", model=asr_model_id, device="cpu")
    add_display_line("ASR Initialized")
