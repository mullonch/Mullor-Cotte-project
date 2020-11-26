from flask import Flask, request, jsonify
# import pickle
import tensorflow as tf
import keras
from keras.models import load_model
import pandas as pd
import numpy as np

server = Flask(__name__)
# Load the model file
# model = pickle.load(open('Model/model.h5', 'rb'))


# load the model, and pass in the custom metric function
# global graph
# graph = tf.get_default_graph()
model = load_model('Model/model.h5')


@server.route('/predict', methods=['POST'])
def predict():
    title = request.json['title']
    date = request.json['date']
    text = request.json['text']
    subject = request.json['subject']

    df = pd.DataFrame(data={"title": [title], "date": [date], "text": [text], "subject": [subject]})
    return jsonify(df)
    #
    # arr = np.array([title, date, text, subject])
    # return arr


def run_request():
    index = int(request.json['index'])
    list_color = ['red', 'green', 'blue', 'yellow', 'black']
    return list_color[index]


@server.route('/', methods=['GET', 'POST'])
def get_color():
    if request.method == 'GET':
        return 'The model is up and running. Send a POST request'
    else:
        return run_request()


@server.route('/hello')
def say_hello():
    return 'hello miss ele'


@server.route('/get', methods=['GET'])
def get():
    return 'test get'


@server.route('/post', methods=['POST'])
def post():
    return run_request()

    # return jsonify(request.json)
#     index = request.json['index']
#    list = ['red', 'green', 'blue', 'yellow', 'black']

#    return index
#    return list[index]
