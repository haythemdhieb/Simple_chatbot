import json
import os, sys
import numpy as np
sys.path.append("/home/haythem/Desktop/Work/training/chatbot/")
from utils import preproces_text

with open("intents.json", "r") as file:
    intents = json.load(file)

all_words = []
tags = []
data = []
for intent in intents["intents"]:
    tag = intent["tag"]
    tags.append(tag)
    for pattern in intent["patterns"]:
        words_from_sentence = preproces_text(pattern)
        all_words.extend(words_from_sentence)
        data.append((words_from_sentence, tag))
#Create training data
X_train=[]
y_train=[]
for (pattern_sentence,tag) in data:
    bag=bag_of_words(pattern_sentence,all_words)
    X_train.append(bag)
    label=tags.index(tag)
    y_train.append(label)

X_train=np.array(X_train)
y_train=np.array(y_train)
