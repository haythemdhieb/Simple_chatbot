
# Simple chatbot implementation with PyTorch.

- The implementation is really simple and straightforward.
- The chat bot model is a Feed Forward Neural net with 3 hidden layers.
- Customisation of this chatbot is easy since we can modify the intents.json file to another use cas.

This implementation was inspired froipm python engineer youtube channel with some improvements on the data processing: [https://www.youtube.com/watch?v=k1SzvvFtl4w]

## Installation
```
We need to install `nltk` first:
 ```console
pip install nltk
 ```

If you get an error during the first run, you also need to install `nltk.tokenize.punkt`:
Run this once in your terminal:
 ```console
$ python
>>> import nltk
>>> nltk.download('punkt')
```

## Usage
Run
```console
python train.py
```
This will dump `data.pth` file. And then run
```console
python chat.py
```
## Customize
Have a look at [intents.json](intents.json). We can customize it according to our own use case. We need to modify the `tag` the `patterns` and the possible `responses`.
```console
{
  "intents": [
    {
      "tag": "greeting",
      "patterns": [
        "Hi",
        "Hey",
        "How are you",
        "Is anyone there?",
        "Hello",
        "Good day"
      ],
      "responses": [
        "Hey :-)",
        "Hello, thanks for visiting",
        "Hi there, what can I do for you?",
        "Hi there, how can I help?"
      ]
    },
    ...
  ]
}
```
## Future improvements
For future improvements, we will use one of BERT'S pretrained models, but the results are quite impressive, we got a loss of `0.0004` and we might design a freindly user internface or create a web app for this chatbot.
