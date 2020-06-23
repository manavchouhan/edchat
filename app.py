import nltk
from flask import Flask, render_template, request
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json
import pickle

class cl:
    model=None
    training=[]
    output=[]

def modeltrain():
    with open("intents.json") as file:
        data = json.load(file)

    try:
        with open("data.pickle", "rb") as f:
            words, labels, cl.training, cl.output = pickle.load(f)
    except:
        words = []
        labels = []
        docs_x = []
        docs_y = []

        for intent in data["intents"]:
            for pattern in intent["patterns"]:
                wrds = nltk.word_tokenize(pattern)
                words.extend(wrds)
                docs_x.append(wrds)
                docs_y.append(intent["tag"])

            if intent["tag"] not in labels:
                labels.append(intent["tag"])

        words = [stemmer.stem(w.lower()) for w in words if w != "?"]
        words = sorted(list(set(words)))

        labels = sorted(labels)

        cl.training = []
        cl.output = []

        out_empty = [0 for _ in range(len(labels))]

        for x, doc in enumerate(docs_x):
            bag = []

            wrds = [stemmer.stem(w.lower()) for w in doc]

            for w in words:
                if w in wrds:
                    bag.append(1)
                else:
                    bag.append(0)

            output_row = out_empty[:]
            output_row[labels.index(docs_y[x])] = 1

            cl.training.append(bag)
            cl.output.append(output_row)


        cl.training = numpy.array(cl.training)
        cl.output = numpy.array(cl.output)

        with open("data.pickle", "wb") as f:
            pickle.dump((words, labels, cl.training, cl.output), f)
    #cl.training.append(1)
    print(cl.training)
    modelst()

    #try:
    #    cl.model.load("model.tflearn")
    #except:
    cl.model.fit(cl.training, cl.output, n_epoch=2000, batch_size=8, show_metric=True)
    cl.model.save("model.tflearn")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)

def modelst():
    tensorflow.reset_default_graph()

    net = tflearn.input_data(shape=[None, len(cl.training[0])])
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, len(cl.output[0]), activation="softmax")
    net = tflearn.regression(net)

    cl.model = tflearn.DNN(net)

def chat(user_response):
    with open("intents.json") as file:
        data = json.load(file)
    with open("data.pickle", "rb") as f:
        words, labels, cl.training, cl.output = pickle.load(f)
    print("Start talking with the bot (type quit to stop)!")

    results = cl.model.predict([bag_of_words(user_response, words)])
    results_index = numpy.argmax(results)
    tag = labels[results_index]

    for tg in data["intents"]:
        if tg['tag'] == tag:
            responses = tg['responses']

    return random.choice(responses)
#modeltrain()

# Keyword Matching
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey",)
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]

def greeting(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

def response(user_response):
    return(chat(user_response))

app = Flask(__name__)


@app.route("/")
def home():
    with open("data.pickle", "rb") as f:
        words, labels, cl.training, cl.output = pickle.load(f)
    modelst()
    cl.model.load("model.tflearn")
    #chat()
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    #userText = request.args.get('msg')
    #return str(english_bot.get_response(userText))
    user_response = request.args.get('msg')
    user_response=user_response.lower()
    if(user_response!='bye'):
        if(user_response=='thanks' or user_response=='thank you' ):
            return str("You are welcome..")
        else:
            #sent_tokens.remove(user_response)
            if(greeting(user_response)!=None):
                return str(greeting(user_response))
            else:
                #print("ROBO: ",end="")
                return str(response(user_response))


    else:
        return str("Bye! take care..")


if __name__ == "__main__":
    app.run(debug = True)
