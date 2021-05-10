import nltk, re
from nltk.corpus import stopwords
import numpy as np
from bs4 import BeautifulSoup
from nltk import word_tokenize, sent_tokenize
from nltk.stem import LancasterStemmer, WordNetLemmatizer


stop_words = list(stopwords.words("english"))


def bag_of_words(sentence, all_words):
    """
    Function that creates the bag of words embeeding vector for a given sentence
    """
    bag = np.zeros(len(all_words), dtype=np.float32)
    for (index, word) in enumerate(sentence):
        if word in all_words:
            bag[all_words.index(word)] = 1
    return bag


def preproces_text(sentence):
    """
    Function that preprocess input sentence: tokenize, lowwer the words and removes speacial characters from each sentence
    """
    soup = BeautifulSoup(sentence, "html.parser")
    sentence = soup.get_text()
    sentence = re.sub("\[[^]]*\]", "", sentence)
    tokenizer = nltk.RegexpTokenizer(r"\w+")
    lemmatizer = WordNetLemmatizer()
    sentence = tokenizer.tokenize(sentence)
    stemmer = LancasterStemmer()
    new_sentence = []
    for word in sentence:
        if word.isnumeric() == False and word.strip() != "":
            new_sentence.append(
                re.sub(
                    r"[^a-zA-Z0-9]+",
                    " ",
                    stemmer.stem(lemmatizer.lemmatize(word.lower())),
                )
            )
    return new_sentence
