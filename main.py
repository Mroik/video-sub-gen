from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess


def example():
    model_dir = "iic/SenseVoiceSmall"
    wav_file = "test.wav"
    
    model = AutoModel(
        model=model_dir,
        vad_model="fsmn-vad",
        vad_kwargs={"max_single_segment_time": 30000},
        device="cuda:0",
    )
    
    # en
    res = model.generate(
        input=wav_file,
        cache={},
        language="ja",  # "zn", "en", "yue", "ja", "ko", "nospeech"
        use_itn=True,
        batch_size_s=60,
        merge_vad=True,  #
        merge_length_s=15,
    )
    text = rich_transcription_postprocess(res[0]["text"])
    print(text)


def detect_segments():
    model = AutoModel(model="fsmn-vad")
    wav_file = "test.wav"
    res = model.generate(input=wav_file)
    print(res)


def main():
    detect_segments()


if __name__ == "__main__":
    main()
