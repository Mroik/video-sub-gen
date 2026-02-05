# Video subtitles generator
This tool can be used to extract and translate subtitles from a video. To see which languages are supported for
extraction from the audio you can refer to https://www.modelscope.cn/models/iic/SenseVoiceSmall, to see which languages
are supported for translation you can refer to https://github.com/Helsinki-NLP/Opus-MT.

To pass the preferred languages use the [ISO 639](https://en.wikipedia.org/wiki/List_of_ISO_639_language_codes) format.

## Dependencies
```sh
pip install -r requirements.txt
```

Depending on the GPU you're using you'll need to install a different version of `pytorch`. Please refer to
https://pytorch.org/get-started/locally/.

I personally use a GTX 1070 so I need to install the pytorch version for CUDA 12.6.
```sh
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
```

## Translation model
By default this project uses a local model for translation, but you can use the Google Cloud Translation API instead by
passing the `-g` argument. To use the Translation API you'll need to setup access to the API on your own through the
Google Cloud Console. Once you've setup the API you can download the client secrets and place them in
`credentials.json`.

> [!WARNING]
> Beware that the Google Cloud Translation API requires you to enable billing. You only get 500000 characters (or 10$
> worth of credit) a month of free interaction with the Translation API.
