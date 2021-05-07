import json
import os,sys
sys.path.append("/home/haythem/Desktop/Work/training/chatbot/")
from utils import preproces_text
with open("intents.json","r") as file:
    intents=json.load(file)

all_words=[]
tags=[]
xy=[]
for intent in intents["intents"]:
    tag=intent["tag"]
    tags.append(tag)
    for pattern in intent["patterns"]:
        words_from_sentence=preproces_text(pattern)
        all_words.extend(words_from_sentence)
        xy.append((words_from_sentence, tag))

for 

 

