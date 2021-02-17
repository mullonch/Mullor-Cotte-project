"""
module de tests fonctionnels
"""

import sys
import os
import pandas as pd
import ../utils

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_fake_article_message():
    """
    test de la fonction message (prédiction = 1)
    """
    assert utils.message(1) == "This is a real news article"


def test_real_article_message():
    """
    test de la fonction message (prédiction = 0)
    """
    assert utils.message(0) == "This is a fake"


def test_remove_between_square_brackets():
    """
    test de la fonction de suppression des crochets
    """
    assert utils.remove_between_square_brackets("this is a [text]") == "this is a "


def test_remove_url():
    """
    test suppression des URLs
    """
    assert utils.remove_url("welcome to https://www.google.com/") == "welcome to "


def test_strip_html():
    """
    test suppression des balises HTML
    """
    assert utils.strip_html("<div><br>Hello</div>") == "Hello"


def test_tokenize():
    """
    test tokenization
    """
    text = ["there is no doubt, this application is awesome"]
    array = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 539, 475, 785, 1642, 183, 4247, 475, 4080]]

    comparison = array == utils.tokenize(text)
    assert comparison.all()


def test_remove_stopwords():
    """
    test suppression des stopwords
    """
    assert utils.remove_stopwords("is it fake news ?") == "fake news"


def test_formate_dataset():
    """
    test formatage dataset
    """
    df_test = pd.DataFrame(
        data={"title": ["1st title", "2nd title"],
              "date": ["1st date", "2nd date"],
              "text": ["1st text", "2nd text"],
              "subject": ["1st subject", "2nd subject"]})
    df_returned = pd.DataFrame(
        data={"title": ["1st title", "2nd title"],
              "text": ["1st text 1st title", "2nd text 2nd title"]})
    assert df_returned.sort_index(
        inplace=True) == utils.formate_dataset(
        df_test).sort_index(inplace=True)


def test_denoise_text():
    """
    test suppression du bruit dans le texte
    """
    assert utils.denoise_text(
        "<div><br>is there fake news on this site https://www.bfmtv.com?</div>") == "fake news site"

def test_prediction():
    """
    test prediction
    """
    dataf = pd.DataFrame(
        data={"title": ["1st title"],
              "date": ["1st date"],
              "text": ["1st text"],
              "subject": ["1st subject"]})
    assert utils.prediction(dataf) in (0, 1)
