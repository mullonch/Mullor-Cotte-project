import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app


def test_fake_article_message():
    assert "This is a real news article" == app.message(1)

def test_real_article_message():
    assert "This is a fake" == app.message(0)
