"""
module utils.py
diverses fonctions utiles
prétraitement des données
"""
import pickle
import re
import string
import os
from bs4 import BeautifulSoup
from keras.models import load_model
from keras.preprocessing import sequence
from nltk.corpus import stopwords
import nltk
import numpy as np

# absolute path to this file
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
# load model
model = load_model(os.path.join(FILE_DIR, 'Model', 'model.h5'))


def prediction(data):
    """
    Apply function for NLP processing and prediction
    """
    data['text'] = data['text'].apply(denoise_text)

    # Tokenize sample
    tokenized_text = tokenize(data['text'].values)

    # transform sample in numpy array and reshape in (1,300)
    sample_np = np.array(tokenized_text).reshape(-1, 300)

    return model.predict_classes(sample_np)[0][0]


def message(pred_result):
    """
    Retourne une phrase décrivant le résultat d'une prédiction
    :param prediction: résultat d'une prédiction
    :return: Texte decrivant le résultat de la prédiction
    """
    assert pred_result in (0, 1), "The model encountered an issue"
    if pred_result:
        return "This is a real news article"
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
    return re.sub('[[][^]]*[]]', '', text)


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


def formate_dataset(dframe):
    """
    Formate le dataset
    :param dframe: dataframe à formater
    :return: dataframe formaté
    """
    dframe['text'] = dframe['text'] + " " + dframe['title']
    del dframe['title']
    del dframe['subject']
    del dframe['date']
    return dframe
