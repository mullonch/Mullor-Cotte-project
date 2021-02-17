"""
    Module app.py
    Application hébergeant l'API développée dans le cadre du projet sopra Valdom 2021
"""

import os
from flask import Flask, request, jsonify, render_template
from flask_swagger_ui import get_swaggerui_blueprint
from keras.models import load_model
import pandas as pd
from utils import formate_dataset, prediction, message

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


##################################### API ##########################################

@server.route('/api/predict', methods=['POST'])
def api_predict():
    """
        Renvoie un texte decrivant le résultat de la prédiction
    """

    assert isinstance(request.json['title'], str), "The title of the article is not defined or not string"
    assert isinstance(request.json['text'], str), "The text of the article is not defined or not string"

    title = request.json['title']
    date = request.json['date']
    text = request.json['text']
    subject = request.json['subject']

    # Create dataframe
    data = formate_dataset(pd.DataFrame(data={"title": [title], "date": [date], "text": [text], "subject": [subject]}))
    # Return prediction
    y = prediction(data)

    return jsonify(message(int(y)))


@server.route('/api/hello')
def api_hello():
    """
    Fonction de test, dit bonjour à l'utilisateur
    """
    return 'Welcome to the real article classifier API !'


@server.route('/api')
def test_api():
    return jsonify({"API status": "running"})


##################################### Inerface web ##########################################

@server.route("/", methods=['GET'])
def home():
    """
    :return: page d'accueil de l'application
    """
    return render_template('home.html')


@server.route('/predict', methods=['POST'])
def predict():
    """
        Renvoie un template decrivant le résultat de la prédiction
    """
    assert isinstance(request.form['title'], str), "The title of the article is not defined or not string"
    assert isinstance(request.form['text'], str), "The text of the article is not defined or not string"

    title = str(request.form["title"])
    date = str(request.form["date"])
    text = str(request.form["text"])
    subject = str(request.form["subject"])

    # Create dataframe
    data = formate_dataset(pd.DataFrame(data={"title": [title], "date": [date], "text": [text], "subject": [subject]}))
    # Return prediction
    y = prediction(data)

    return render_template('result.html', prediction=y, title=title)
