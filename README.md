If you're using a Pascal NVidia GPU you'll need the 12.6 bindings for CUDA
```sh
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
```

The translation happens through the Google Cloud Translation API, note that this API requires you to enable billing. You
do get 10$ a month of free credit for this API (500000 characters worth of translation). The first time you activate
billing google gives you 250€ worth of credit to use over 90 days.
