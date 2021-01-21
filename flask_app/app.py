
import pickle
import re
import string

import flask
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

from flask import Flask, request, jsonify, Response, send_from_directory, Blueprint

from flask_swagger_ui import get_swaggerui_blueprint


# import pickle
from keras.models import load_model
from keras.preprocessing import sequence
from nltk.corpus import stopwords
import nltk
import pandas as pd
from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer
import numpy as np


server = Flask(__name__)

# Load the model file
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


# Load the model file
model = load_model('Model/model.h5')


@server.route('/predict', methods=['POST'])
def predict():
    title = request.json['title']
    date = request.json['date']
    text = request.json['text']
    subject = request.json['subject']
    # Create dataframe
    df = pd.DataFrame(data={"title": [title], "date": [date], "text": [text], "subject": [subject]})

    data = formate_dataset(df)

    # Apply function for NLP processing
    data['text'] = data['text'].apply(denoise_text)

    # Tokenize sample
    tokenized_text = tokenize(data['text'].values)

    # transform sample in numpy array and reshape in (1,300)
    sample_np = np.array(tokenized_text).reshape(-1, 300)

    # load model and predict
    model = load_model('Model/model.h5')
    prediction = model.predict_classes(sample_np)[0][0]

    return jsonify(message(int(prediction)))


@server.route('/hello')
def say_hello():
    return 'Welcome to the real article classifier !'


# predict with train data prepared retrieved in the notebook
@server.route('/predict_with_data')
def predict_with_data():
    print("ok")
    data_train = pd.read_csv('Model/train_data.csv')
    for i in range(10):
        sample = data_train.values.tolist()[i][1:]  # remove index added by pandas
        sample_np = np.array(sample)

        prediction = model.predict_classes(sample_np.reshape(-1, 300))[0][0]

    return jsonify(prediction)


def message(prediction):
    if prediction == 1:
        return "This is a real news article"
    else:
        return "This is a fake"


def tokenize(text):
    # loading
    with open('Tokenizer/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    tokenized_text = tokenizer.texts_to_sequences(text)
    return sequence.pad_sequences(tokenized_text, maxlen=300)


def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()


# Removing the square brackets
def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)


# Removing URL's
def remove_between_square_brackets(text):
    return re.sub(r'http\S+', '', text)


# Removing the stopwords from text
def remove_stopwords(text):
    nltk.download("stopwords")
    stop = set(stopwords.words('english'))
    punctuation = list(string.punctuation)
    stop.update(punctuation)
    final_text = []
    for i in text.split():
        if i.strip().lower() not in stop:
            final_text.append(i.strip())
    return " ".join(final_text)


# Removing the noisy text
def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    text = remove_stopwords(text)
    return text


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
