![Build Status](https://github.com/VALDOM-PROJET-TRANSVERSE-2020/Mullor-Cotte-project/Mullor-Cotte-project/workflows/python-app.yml/badge.svg)

<img src="https://can7.fr/images/inp-enseeiht.jpg" width=300, style="max-width: 110px; display: inline" alt="N7"/>

# Real or fake news ?

The aim of this project is to run in production a deep learning model supposed to detect fake news. The model can be 
deployed in a simple API thanks to the Flask web-framework. 

<div align="center">
<img src="https://gosint.files.wordpress.com/2018/07/fake-news-vs-truth.jpg?w=768&h=445" width=500, style="display: block; margin-left: auto; margin-right: auto; text-align:center;" alt="notebook_img"/>
</div>

### Why fake news is a problem ?

Fake news refers to misinformation, disinformation or mal-information which is spread through word of mouth and traditional
media and more recently through digital forms of communication such as edited videos, memes, unverified advertisements and 
social media propagated rumours.Fake news spread through social media has become a serious problem, with the potential of 
it resulting in mob violence, suicides etc... as a result of misinformation circulated on social media.

### Technologies used in this project

**Model**

The deep learning model used was retrieved in a notebook found in kaggle : https://www.kaggle.com/madz2000/nlp-using-glove-embeddings-99-87-accuracy.
This model is using word embedding thanks to a pretrained Glove model. As NLP model used in dataset containing several 
sentences, this model is a RNN using LSTM layers which stands for Long short-term memory.

**API**

After being retrained, this model had to be runned in production in a Flask API well documented thanks to the interface 
description langage Swaggger. 

**Working environment**

The Flask API was coded in Python langage with a virtual environment. To use this project, it is highly recommended to 
create a virtual environment and get the necessary libraries :
```
pip install -r requirements.txt
```

To serve this little API, a Gunicorn application server and a Nginx web server had to be set up. The application server 
is used to communicate with the application and the web server is used to treat HTTP requests in our API.

This servers and the Flask API are gathered in a Docker container. You can easily use this project cloning this repository 
and build this container (if you have Docker, Gunicorn and Nginx previously installed) :
```
git clone https://github.com/VALDOM-PROJET-TRANSVERSE-2020/Mullor-Cotte-project.git
./run_docker.sh
```


