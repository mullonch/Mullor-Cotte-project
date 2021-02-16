import pickle
import re
import string
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify, Response, send_from_directory, Blueprint, render_template
from flask_swagger_ui import get_swaggerui_blueprint
from keras.models import load_model
from nltk.corpus import stopwords
import nltk
import pandas as pd
from keras.preprocessing import text, sequence
import numpy as np
import os

server = Flask(__name__)

SWAGGER_URL = '/swagger'
API_URL = '/static/swagger.json'
swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': 'Projet Transverse'
    }
)
server.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

# absolute path to this file
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
# load model
# model = load_model('Model/model.h5')
model = load_model(os.path.join(FILE_DIR, 'Model', 'model.h5'))


@server.route('/predict', methods=['POST'])
def predict():
    assert isinstance(request.json['title'], str), "The title of the article is not defined or not string"
    assert isinstance(request.json['text'], str), "The text of the article is not defined or not string"

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

    prediction = model.predict_classes(sample_np)[0][0]

    return jsonify(message(int(prediction)))


@server.route('/hello')
def say_hello():
    # if request.headers['Content-Type'] == 'application/json; charset=UTF-8':
    #     return 'Welcome to the real article classifier !'
    # else:
    #     return "NOT API"
    html = False
    if not request.json or not 'html' in request.json:
        html=False
        return 'Welcome to the real article classifier !'
    elif request.json['html'] == "True":
        html = True
        return render_template('welcome.html')

    # return 'Welcome to the real article classifier !' + str(html)

    # if request.args['type'] == 'json':
    #     return 'Welcome to the real article classifier !'
    # else:
    #     return "<html>\
    #           <body>\
    #             <strong>Welcome to the real article classifier !</strong>\
    #           </body>\
    #         </html>"


# predict with train data prepared retrieved in the notebook
# @server.route('/predict_with_data')
# def predict_with_data():
#    print("ok")
#    data_train = pd.read_csv('Model/train_data.csv')
#    for i in range(10):
#        sample = data_train.values.tolist()[i][1:]  # remove index added by pandas
#        sample_np = np.array(sample)
#         prediction = model.predict_classes(sample_np.reshape(-1, 300))[0][0]
#
#    return jsonify(prediction)


def message(prediction):
    assert (prediction == 0 or prediction == 1), "The model encountered an issue"
    if prediction == 1:
        return "This is a real news article"
    elif prediction == 0:
        return "This is a fake"


def tokenize(text):
    # loading tokenizer file
    # with open('Tokenizer/tokenizer.pickle', 'rb') as handle:
    with open(os.path.join(FILE_DIR, 'Tokenizer', 'tokenizer.pickle'), 'rb') as handle:
        tokenizer = pickle.load(handle)

    tokenized_text = tokenizer.texts_to_sequences(text)
    return sequence.pad_sequences(tokenized_text, maxlen=300)


def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()


# Removing all between the square brackets
def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)


# Removing URL's
def remove_url(text):
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
    text = remove_url(text)
    text = remove_stopwords(text)
    return text


def formate_dataset(df):
    df['text'] = df['text'] + " " + df['title']
    del df['title']
    del df['subject']
    del df['date']
    return df