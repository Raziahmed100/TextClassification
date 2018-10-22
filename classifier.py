import pandas as pd
dataset = pd.read_csv('data.csv', encoding='ISO-8859-1');

import re
import nltk

nltk.download('punkt')
from nltk.tokenize import word_tokenize as wt 

nltk.download('stopwords')
from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

#spell correction
from autocorrect import spell

data = []

for i in range(dataset.shape[0]):
    sms = dataset.iloc[i, 1]

    # remove non alphabatic characters
    sms = re.sub('[^A-Za-z]', ' ', sms)

    # make words lowercase, because Go and go will be considered as two words
    sms = sms.lower()

    # tokenising
    tokenized_sms = wt(sms)

    # remove stop words and stemming
 
    row = []
crimefile = open(fileName, 'r')
for line in crimefile.readlines():
    row.append([line])
    for i in line.split(","):
        row[-1].append(i)

# creating the feature matrix 
from sklearn.feature_extraction.text import CountVectorizer
matrix = CountVectorizer(max_features=1000)
X = matrix.fit_transform(data).toarray()
y = dataset.iloc[:, 0]

# split train and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Naive Bayes 
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import  cross_val_score
import time
from sklearn.datasets import  load_iris

iris = load_iris()

models = [GaussianNB(), DecisionTreeClassifier(), SVC()]
names = ["Naive Bayes", "Decision Tree", "SVM"]
for model, name in zip(models, names):
    print name
    start = time.time()
    for score in ["accuracy", "precision", "recall"]:
        print score,
        print " : ",
        print cross_val_score(model, iris.data, iris.target,scoring=score, cv=10).mean()
    print time.time() - start

# predict class
y_pred = classifier.predict(X_test)

# Confusion matrix
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)

accuracy = accuracy_score(y_test, y_pred)


