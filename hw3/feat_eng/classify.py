#!/usr/bin/env python
#
#
# Starting code from:
# http://www.cs.colorado.edu/~jbg/teaching/CSCI_5622/04.py
# https://github.com/ezubaric/ml-hw/blob/master/feat_eng/classify.py
# 
# 
# Modified By:
# Paul Boschert <paul@boschert.net>, <paul.boschert@colorado.edu>
# CSCI 5622 - Machine Learning: Homework 3
#                                                                                                                         

from csv import DictReader, DictWriter

import numpy as np
from numpy import array

# TODO look into these libraries
#from sklearn.metrics import confusion_matrix
#from sklearn.feature_extraction.text import HashingVectorizer
#from sklearn.metrics import accuracy_score
#from nltk.corpus import wordnet as wn
#from nltk.corpus import brown
#from nltk.util import ngrams
from sklearn.metrics import accuracy_score

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier

kTARGET_FIELD = 'spoiler'
kTEXT_FIELD = 'sentence'


class Featurizer:
    def __init__(self):
        self.vectorizer = CountVectorizer()

    def train_feature(self, examples):
        return self.vectorizer.fit_transform(examples)

    def test_feature(self, examples):
        return self.vectorizer.transform(examples)

    def show_top10(self, classifier, categories):
        feature_names = np.asarray(self.vectorizer.get_feature_names())
        for i, category in enumerate(categories):
            top10 = np.argsort(classifier.coef_[i])[-10:]
            print("%s: %s" % (category, " ".join(feature_names[top10])))

if __name__ == "__main__":

    # Cast to list to keep it all in memory
    train = list(DictReader(open("../data/spoilers/train.csv", 'r')))
    test = list(DictReader(open("../data/spoilers/test.csv", 'r')))

    feat = Featurizer()

    labels = []
    for line in train:
        if not line[kTARGET_FIELD] in labels:
            labels.append(line[kTARGET_FIELD])

    x_train = feat.train_feature(x[kTEXT_FIELD] for x in train)
    x_test = feat.test_feature(x[kTEXT_FIELD] for x in test)

    y_train = array(list(labels.index(x[kTARGET_FIELD]) for x in train))

    # Train classifier
    lr = SGDClassifier(loss='log', penalty='l2', shuffle=True)
    lr.fit(x_train, y_train)

    feat.show_top10(lr, labels) # show the top 10 features used for classification

    # show the accuracy
    train_predictions = lr.predict(x_train)
    print("Training Accuracy: %f" % accuracy_score(y_train, train_predictions))

    predictions = lr.predict(x_test)
    o = DictWriter(open("predictions.csv", 'w'), ["id", "cat"])
    o.writeheader()
    for ii, pp in zip([x['id'] for x in test], predictions):
        d = {'id': ii, 'cat': labels[pp]}
        o.writerow(d)