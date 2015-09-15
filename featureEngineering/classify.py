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


#from nltk.util import ngrams
from sklearn.metrics import accuracy_score
import argparse
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from sklearn.cross_validation import train_test_split
import re




from csv import DictReader, DictWriter

import numpy as np
from numpy import array

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier

kTEXT_FIELD = 'sentence'
kTARGET_FIELD = 'spoiler'
kVERB_FIELD = 'verb'
kPAGE_FIELD = 'page'
kTROPE_FIELD = 'trope'

def camelCaseConvert(value):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1 \2', value)
    return re.sub('([a-z0-9])([A-Z])', r'\1 \2', s1).lower()

class Analyzer:
    def __init__(self, word):
        self.word = word

    def __call__(self, feature_string):
        feats = features_string.split()

        if self.word:
            yield feats[0]

class Featurizer:
    def __init__(self, analyzer = None):
        if analyzer is None:
            print("WARNING: analyzer not given, the number of features will be equal to the "
                  "vocabulary size found by analyzing the data.")

        self.vectorizer = CountVectorizer(analyzer,
                                          stop_words = "english",
                                          strip_accents = "ascii",
                                          ngram_range = (1, 2))

    def train_feature(self, examples):
        return self.vectorizer.fit_transform(examples)

    def test_feature(self, examples):
        return self.vectorizer.transform(examples)

    def show_top10(self, classifier, categories):
        print("\nTop 10 features used for each classification")
        print("--------------------------------------------")
        feature_names = np.asarray(self.vectorizer.get_feature_names())
        if len(categories) == 2:
            top10 = np.argsort(classifier.coef_[0])[-10:]
            bottom10 = np.argsort(classifier.coef_[0])[:10]
            print("Pos: %s" % " ".join(feature_names[top10]))
            print("Neg: %s" % " ".join(feature_names[bottom10]))
        else:
            for i, category in enumerate(categories):
                top10 = np.argsort(classifier.coef_[i])[-10:]
                print("%s: %s" % (category, " ".join(feature_names[top10])))

if __name__ == "__main__":
    # initialize the argument parser and define the arguments
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--word', default=False, action="store_true",
                        help="Use the words of the text as features")
    parser.add_argument('--page', default=False, action="store_true",
                        help="Use the page as a feature, this appears to be the movie/show that the quote is about")
    parser.add_argument('--trope', default=False, action="store_true",
                        help="Use the trope as a feature, this appears to be the author of the quote")
    parser.add_argument('--split', default=False, action="store_true",
                        help="Use the trope as a feature, this appears to be the movie/show that the quote is about")
    parser.add_argument('--genre', default=False, action="store_true",
                        help="Add the genre of the based on the trope")

    flags = parser.parse_args()

    """
    Read in data
    """
    train = list(DictReader(open("../data/spoilers/train.csv", 'r')))
    test = list(DictReader(open("../data/spoilers/test.csv", 'r')))

    labels = []
    for line in train:
        if not line[kTARGET_FIELD] in labels:
            labels.append(line[kTARGET_FIELD])

    #print("Label set: %s" % str(labels))

    
    if flags.trope and flags.page:
        x_train_all = array(list(' '.join((x[kTEXT_FIELD], x[kTROPE_FIELD], x[kPAGE_FIELD])) for x in train))
    elif flags.trope:
        x_train_all = array(list(' '.join((x[kTEXT_FIELD], x[kTROPE_FIELD])) for x in train))
    elif flags.page:
        x_train_all = array(list(' '.join((x[kTEXT_FIELD], x[kPAGE_FIELD])) for x in train))
    else:
        x_train_all = array(list((x[kTEXT_FIELD] for x in train)))

    y_train_all = array(list(labels.index(x[kTARGET_FIELD]) for x in train))

    # since we don't have y values for our testing set, get an approximation for our testing
    # set by splitting our training set in two and using the first part to classify and the
    # second part to validate the accuracy
    if flags.split:
        x_train, x_validate, y_train, y_validate = train_test_split(x_train_all, y_train_all,
                                                                    random_state = 1)
    else:
        x_train = x_train_all
        y_train = y_train_all
        x_validate = []
        y_validate = []

    print("Size of training data set: %s" % len(y_train))
    print("Size of validation data set: %s" % len(y_validate))

    # Get features
    analyzer = Analyzer(flags.word)

    feat = Featurizer(analyzer)

    x_train = feat.train_feature(x_train)

    if flags.split:
        # deal with validation test data
        x_validate = feat.test_feature(x_validate)

    # deal with the test data set
    if flags.trope and flags.page:
        x_test = array(list(' '.join((x[kTEXT_FIELD], x[kTROPE_FIELD], x[kPAGE_FIELD])) for x in test))
    elif flags.trope:
        x_test = array(list(' '.join((x[kTEXT_FIELD], x[kTROPE_FIELD])) for x in test))
    elif flags.page:
        x_test = array(list(' '.join((x[kTEXT_FIELD], x[kPAGE_FIELD])) for x in test))
    else:
        x_test = array(list((x[kTEXT_FIELD] for x in test)))

    x_test = feat.test_feature(x_test)

    # Train classifier
    lr = SGDClassifier(loss='log', penalty='l2', shuffle=True)
    lr.fit(x_train, y_train)

    feat.show_top10(lr, labels) # show the top 10 features used for classification

    # show the training accuracy
    print("  Training Accuracy: %f" % accuracy_score(y_train, lr.predict(x_train)))

    if flags.split:
        # show the validation accuracy
        print("Validation Accuracy: %f" % accuracy_score(y_validate, lr.predict(x_validate)))

    # write out the predicitons
    predictions = lr.predict(x_test)
    o = DictWriter(open("predictions.csv", 'w'), ["id", "spoiler"])
    o.writeheader()
    for ii, pp in zip([x['id'] for x in test], predictions):
        d = {'id': ii, 'spoiler': labels[pp]}
        o.writerow(d)

