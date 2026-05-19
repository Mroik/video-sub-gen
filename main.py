# TODO: Make use of timestamp prediction

import argparse
from abc import ABC, abstractmethod
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from moviepy import VideoFileClip
from os import remove, environ
from os.path import exists
from typing import Tuple, List
from html import unescape


API_KEY = environ.get("GOOGLE_API")
SOURCE_LANG = "ja"
TARGET_LANG = "en"
MODEL = "iic/SenseVoiceSmall"
SRT_PATH = "sub.srt"
TRANSLATOR_MODEL = "opus-mt"
TRANSLATION_MODE = None


class Translator(ABC):
    @abstractmethod
    def translate(self, text: str, source: str, target: str):
        ...


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


class SelfHostedTranslator(Translator):
    def __init__(self):
        import nltk
        from easynmt import EasyNMT
        nltk.download("punkt_tab")
        self.model = EasyNMT(TRANSLATOR_MODEL)

    def translate(self, text: str, source: str, target: str):
        res = self.model.translate(text, target_lang=target, source_lang=source)
        return unescape(res)


class GoogleTranslator(Translator):
    def __init__(self):
        from google.cloud import translate_v2 as translate
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
        from google_auth_oauthlib.flow import InstalledAppFlow

        creds = None
        if exists("token.json"):
            creds = Credentials.from_authorized_user_file("token.json")

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    "credentials.json",
                    "https://www.googleapis.com/auth/cloud-translation",
                )
                creds = flow.run_local_server(port=0)

            with open("token.json", "w") as token:
                token.write(creds.to_json())

        self.client = translate.Client(credentials=creds)

    def translate(self, text: str, source: str, target: str) -> str:
        res = self.client.translate(text, source_language=source, target_language=target)
        return unescape(res["translatedText"])


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
    sec = sec % 60
    minutes = minutes % 60
    return f"{hours:02}:{minutes:02}:{sec:02},{mil:03}"


def main(args):
    wav_path = "test.wav"
    video_path = args.input_video
    srt_path = args.o

    extract_wav(video_path, wav_path)

    segments = detect_segments(wav_path)
    seg_with_stamp: List[Tuple[List[int], str]] = []
    video = VideoFileClip(video_path)
    model = Transcriber()
    interval: List[int] = None
    for i, interval in enumerate(segments):
        seg_filename = f"{i}_{wav_path}"

        audio_seg = video.subclipped(interval[0] / 1000, interval[1] / 1000).audio
        audio_seg.write_audiofile(seg_filename)

        text = model.transcribe(seg_filename, SOURCE_LANG)
        remove(seg_filename)

        seg_with_stamp.append((interval, text))

    remove(wav_path)
    del model

    translator: Translator = TRANSLATION_MODE()
    seg_with_stamp = map(lambda x: (x[0], translator.translate(x[1], SOURCE_LANG, TARGET_LANG)), seg_with_stamp)

    print("Starting translation...")
    with open(srt_path, "w") as fd:
        for i, (interval, text) in enumerate(seg_with_stamp, start=1):
            fd.write(f"{i}\n")
            fd.write(timestamp_format(interval[0]))
            fd.write(" --> ")
            fd.write(f"{timestamp_format(interval[1])}\n")
            fd.write(f"{text}\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_video")
    parser.add_argument("-o", type=str, default=SRT_PATH)
    parser.add_argument("-g", action="store_true", help="Use the Google Cloud Translation API instead of the local "
                        "translator")
    parser.add_argument("-s", default="ja", help="The language from which to translate")
    parser.add_argument("-t", default="en", help="The language to translate to")
    args = parser.parse_args()
    TRANSLATION_MODE = GoogleTranslator if args.g else SelfHostedTranslator
    if args.s:
        SOURCE_LANG = args.s
    if args.t:
        TARGET_LANG = args.t
    main(args)
