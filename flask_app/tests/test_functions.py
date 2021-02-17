import pytest
import sys
import os
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app
import utils


def test_fake_article_message():
    assert "This is a real news article" == utils.message(1)


def test_real_article_message():
    assert "This is a fake" == utils.message(0)


def test_remove_between_square_brackets():
    assert "this is a " == utils.remove_between_square_brackets("this is a [text]")


def test_remove_url():
    assert "welcome to " == utils.remove_url("welcome to https://www.google.com/")


def test_strip_html():
    assert "Hello" == utils.strip_html("<div><br>Hello</div>")


def test_tokenize():
    text = ["there is no doubt, this application is awesome"]
    array = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 539, 475, 785, 1642, 183, 4247, 475, 4080]]

    comparison = array == utils.tokenize(text)
    assert comparison.all() == True


def test_remove_stopwords():
    assert "fake news" == utils.remove_stopwords("is it fake news ?")


def test_formate_dataset():
    df_test = pd.DataFrame(
        data={"title": ["1st title", "2nd title"], "date": ["1st date", "2nd date"], "text": ["1st text", "2nd text"],
              "subject": ["1st subject", "2nd subject"]})
    df_returned = pd.DataFrame(
        data={"title": ["1st title", "2nd title"], "text": ["1st text 1st title", "2nd text 2nd title"]})
    assert df_returned.sort_index(inplace=True) == utils.formate_dataset(df_test).sort_index(inplace=True)


def test_denoise_text():
    assert "fake news site" == utils.denoise_text("<div><br>is there fake news on this site https://www.bfmtv.com?</div>")
