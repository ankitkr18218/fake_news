from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import numpy as np


from sklearn.feature_extraction.text import TfidfVectorizer

app=Flask(__name__)

model=pickle.load(open('model.pkl', 'rb'))
v=pickle.load(open("vectorizer.pickle", 'rb'))

# preprocessing
import nltk
import re
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem import PorterStemmer
porter=PorterStemmer()
stop_word=stopwords.words('english')
tokenized_sents=[]
def preprocessing(text):
  data=[]
  for i in text:
    sent=re.sub(r'[^a-zA-Z]', ' ', str(i))
    sent=sent.lower()
    sent_token=sent.split()
    sent_stem=[]
    for j in sent_token:
      d=porter.stem(j)
      if d not in stop_word:
        sent_stem.append(d)
    tokenized_sents.append(sent_stem)
    sent=' '.join(sent_stem)
    data.append(sent)
  return data

@app.route('/')
def home():
  return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
  ## prediction code
  sent_stem=[str(x) for x in request.form.values()]
  sent=' '.join(sent_stem)
  #print(sent)

  testing = {"title": [sent]}
  test_sent = pd.DataFrame(testing)
  d = preprocessing(test_sent["title"])
  #print(d)
  new_test_sent = v.transform(d)
  value_predicted=model.predict(new_test_sent)
  if value_predicted==0:
    prediction="fake"
  else:
    prediction="real"
  return render_template('index.html', predicted='This news is predicted as {}'.format(prediction))

if __name__=="__main__":
  app.run(debug=True)