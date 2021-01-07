from flask import Flask, request, jsonify, Response, send_from_directory, Blueprint

from flask_swagger_ui import get_swaggerui_blueprint

# import pickle
import tensorflow as tf
import keras
from keras.models import load_model
import pandas as pd
from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer
import numpy as np

server = Flask(__name__)
# Load the model file
# model = pickle.load(open('Model/model.h5', 'rb'))


# load the model, and pass in the custom metric function
# global graph
# graph = tf.get_default_graph()
model = load_model('Model/model.h5')


@server.route('/static/<path:path>', methods=['GET'])
def send_static(path):
    return send_from_directory('static', path)


SWAGGER_URL = '/swagger'
API_URL = '/static/swagger.json'
swaggerui_blueprint= get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name':'Projet Transverse'
    }
)
server.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)
server.register_blueprint(Blueprint('request_api', __name__))

@server.route('/predict', methods=['POST'])
def predict():
    title = request.json['title']
    date = request.json['date']
    text = request.json['text']
    subject = request.json['subject']

    df = pd.DataFrame(data={"title": [title], "date": [date], "text": [text], "subject": [subject]})

    dataset = formate_dataset(df)

    # data = str(model.predict(dataset))

    # X = tokenizer(dataset)

    return jsonify("X")
    # return X.to_html(header="true", table_id="table")

    #
    # arr = np.array([title, date, text, subject])
    # return arr


def tokenizer(data_text):
    tokenizer = text.Tokenizer(num_words=10000)
    tokenized_test = tokenizer.texts_to_sequences(data_text)
    X_test = sequence.pad_sequences(tokenized_test, maxlen=300)
    return (X_test)


def formate_dataset(df):
    df['text'] = df['text'] + " " + df['title']
    del df['title']
    del df['subject']
    del df['date']

    return df


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
    return 'hello frero'


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
