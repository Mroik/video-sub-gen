# TODO: Make use of timestamp prediction
# TODO: Add a translator model to translate before generating sub file

import argparse
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from moviepy import VideoFileClip
from os import remove, environ
from os.path import exists
from itertools import count
from typing import Tuple, List
from google.cloud import translate_v2 as translate
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from html import unescape


API_KEY = environ.get("GOOGLE_API")
JAPANESE = "ja"
ENGLISH = "en"
MODEL = "iic/SenseVoiceSmall"
SRT_PATH = "sub.srt"


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


class Translator:
    def __init__(self):
        creds = None
        # The file token.json stores the user's access and refresh tokens, and is
        # created automatically when the authorization flow completes for the first
        # time.
        if exists("token.json"):
            creds = Credentials.from_authorized_user_file("token.json")
        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    "credentials.json",
                    "https://www.googleapis.com/auth/cloud-translation",
                )
                creds = flow.run_local_server(port=0)
            # Save the credentials for the next run
            with open("token.json", "w") as token:
                token.write(creds.to_json())

        self.client = translate.Client(credentials=creds)

    def translate(self, text, source_lang: str, target_lang: str) -> str:
        res = self.client.translate(text, source_language=source_lang, target_language=target_lang)
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
    return f"{hours:02}:{minutes:02}:{sec:02},{mil:03}"


def main(args):
    wav_path = "test.wav"
    video_path = args.input_video
    srt_path = args.o

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

    remove(wav_path)
    del model

    translator = Translator()
    seg_with_stamp = map(lambda x: (x[0], translator.translate(x[1], JAPANESE, ENGLISH)), seg_with_stamp)

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
    args = parser.parse_args()
    main(args)
