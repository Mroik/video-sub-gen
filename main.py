# TODO: Make use of timestamp prediction
# TODO: Add a translator model to translate before generating sub file

from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from moviepy import VideoFileClip
from os import remove
from itertools import count
from typing import Tuple, List


JAPANESE = "ja"
MODEL = "iic/SenseVoiceSmall"


class Transcriber:
    def __init__(self):
        self.model = AutoModel(
            model=MODEL,
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 30000},
            device="cuda:0",
        )

    def transcribe(self, wav_path: str, lang: str) -> str:
        res = self.model.generate(
            wav_path,
            language=lang,
            # use_itn=True,
            # batch_size_s=60,
            # merge_vad=True,  #
            # merge_length_s=15,
        )
        return rich_transcription_postprocess(res[0]["text"])


def detect_segments(wav_path: str) -> List[List[int]]:
    model = AutoModel(model="fsmn-vad")
    res = model.generate(input=wav_path)
    return res[0]["value"]


def extract_wav(video_path: str, wav_path: str):
    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(wav_path)
    audio.close()
    video.close()


def timestamp_format(milli: int) -> str:
    mil = milli % 1000
    sec = milli // 1000
    minutes = sec // 60
    hours = minutes // 60
    return f"{hours:02}:{minutes:02}:{sec:02},{mil:03}"


def main():
    video_path = "test.mp4"
    wav_path = "test.wav"
    srt_path = "sub.srt"

    extract_wav(video_path, wav_path)

    segments = iter(detect_segments(wav_path))
    seg_with_stamp: List[Tuple[List[int], str]] = []
    video = VideoFileClip(video_path)
    model = Transcriber()
    interval: List[int] = None
    for i in count(1):
        seg_filename = f"{i}_{wav_path}"

        try:
            interval = next(segments)
        except StopIteration:
            break

        audio_seg = video.subclipped(interval[0] / 1000, interval[1] / 1000).audio
        audio_seg.write_audiofile(seg_filename)

        text = model.transcribe(seg_filename, JAPANESE)
        remove(seg_filename)

        seg_with_stamp.append((interval, text))

    print(seg_with_stamp)
    remove(wav_path)

    with open(srt_path, "w") as fd:
        for i, (interval, text) in enumerate(seg_with_stamp, start=1):
            fd.write(f"{i}\n")
            fd.write(timestamp_format(interval[0]))
            fd.write(" --> ")
            fd.write(f"{timestamp_format(interval[1])}\n")
            fd.write(f"{text}\n\n")


if __name__ == "__main__":
    main()
