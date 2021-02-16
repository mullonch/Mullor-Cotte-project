"""
    Module app.py
    Application hébergeant l'API développée dans le cadre du projet sopra Valdom 2021
"""

import pickle
import re
import string
import os
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify, render_template
from flask_swagger_ui import get_swaggerui_blueprint
from keras.models import load_model
from keras.preprocessing import sequence
from nltk.corpus import stopwords
import nltk
import pandas as pd
import numpy as np

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
    """
        Renvoie un texte decrivant le résultat de la prédiction
    """
    if not request.json:
        title = str(request.form["title"])
        date = str(request.form["date"])
        text = str(request.form["text"])
        subject = str(request.form["subject"])


    else:
        assert isinstance(request.json['title'], str), "The title of the article is not defined or not string"
        assert isinstance(request.json['text'], str), "The text ogit add .f the article is not defined or not string"

        title = request.json['title']
        date = request.json['date']
        text = request.json['text']
        subject = request.json['subject']

    # Create dataframe
    data = formate_dataset(pd.DataFrame(data={"title": [title], "date": [date], "text": [text], "subject": [subject]}))

    # Apply function for NLP processing
    data['text'] = data['text'].apply(denoise_text)

    # Tokenize sample
    tokenized_text = tokenize(data['text'].values)

    # transform sample in numpy array and reshape in (1,300)
    sample_np = np.array(tokenized_text).reshape(-1, 300)

    prediction = model.predict_classes(sample_np)[0][0]
    if not request.json:
        return render_template('predict.html', prediction=prediction, title=title)
    return jsonify(message(int(prediction)))


@server.route('/hello')
def say_hello():
    """
    Fonction de test, dit bonjour à l'utilisateur / page d'accueil
    """
    if not request.json or not 'api' in request.json:
        return render_template('welcome.html')
    return 'Welcome to the real article classifier !'


def message(prediction):
    """
    Retourne une phrase décrivant le résultat d'une prédiction
    :param prediction: résultat d'une prédiction
    :return: Texte decrivant le résultat de la prédiction
    """
    assert prediction in (0, 1), "The model encountered an issue"
    if prediction == 1:
        return "This is a real news article"
    else:
        return "This is a fake"


def tokenize(text):
    """
    Tokenize le texte
    :param text: texte à transformer
    :return: texte transformé
    """
    # loading tokenizer file
    # with open('Tokenizer/tokenizer.pickle', 'rb') as handle:
    with open(os.path.join(FILE_DIR, 'Tokenizer', 'tokenizer.pickle'), 'rb') as handle:
        tokenizer = pickle.load(handle)

    tokenized_text = tokenizer.texts_to_sequences(text)
    return sequence.pad_sequences(tokenized_text, maxlen=300)


def strip_html(text):
    """
    Supprime le contenu HTML (balises)
    :param text: texte à transformer
    :return: texte transformé
    """
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()


# Removing all between the square brackets
def remove_between_square_brackets(text):
    """
    Supprime tout le contenu se trouvant entre cochets
    :param text: texte à transformer
    :return: texte transformé
    """
    return re.sub('\[[^]]*\]', '', text)


# Removing URL's
def remove_url(text):
    """
    Supprime les URLs
    :param text: texte à transformer
    :return: texte transformé
    """
    return re.sub(r'http\S+', '', text)


# Removing the stopwords from text
def remove_stopwords(text):
    """
    Supprime les mots inutiles (stopwords)
    :param text: texte à transformer
    :return: texte transformé
    """
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
    """
    Applique tous les nettoyages au texte passé en parametre
    :param text: texte à nettoyer
    :return: texte nettoyé
    """
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    text = remove_url(text)
    text = remove_stopwords(text)
    return text


def formate_dataset(df):
    """
    Formate le dataset
    :param df: dataframe à formater
    :return: dataframe formaté
    """
    df['text'] = df['text'] + " " + df['title']
    del df['title']
    del df['subject']
    del df['date']
    return df
