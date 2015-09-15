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


# TODO look into these libraries
#from sklearn.metrics import confusion_matrix
#from sklearn.feature_extraction.text import HashingVectorizer
#from sklearn.metrics import accuracy_score
#from nltk.corpus import wordnet as wn
#from nltk.corpus import brown
#from nltk.util import ngrams
from sklearn.metrics import accuracy_score
import argparse
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize




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


class Analyzer:
    def __init__(self, word, page, trope):
        self.word = word
        self.page = page
        self.trope = trope

    def __call__(self, feature_string):
        feats = features_string.split()

        if self.word:
            yield feats[0]

        if self.page:
            for i in [x for x in feats if x.startswith("P:")]:
                yield i

        if self.trope:
            for i in [x for x in feats if x.startswith("T:")]:
                yield i


class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, sentence):
        return [self.wnl.lemmatize(t) for t in word_tokenize(sentence)]

class Featurizer:
    def __init__(self, analyzer = None):
        if analyzer is None:
            print("WARNING: analyzer not given, the number of features will be equal to the "
                  "vocabulary size found by analyzing the data.")

        self.vectorizer = CountVectorizer(analyzer,
                                          stop_words = "english",
                                          strip_accents = "ascii",
                                          tokenizer = LemmaTokenizer(),
                                          ngram_range = (1, 2))

    def train_feature(self, examples):
        return self.vectorizer.fit_transform(examples)

    def test_feature(self, examples):
        return self.vectorizer.transform(examples)

    def show_top10(self, classifier, categories):
        print("Top 10 features used for each classification")
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
                        help="Use the page as a feature, this appears to be the author of the quote")
    parser.add_argument('--trope', default=False, action="store_true",
                        help="Use the trope as a feature, this appears to be the movie/show that the quote is about")

    flags = parser.parse_args()


    # Cast to list to keep it all in memory
    train = list(DictReader(open("../data/spoilers/train.csv", 'r')))
    # since we don't have y values for our testing set, get an approximation for our testing
    # set by splitting our training set in two and using the first half to classify and the
    # second half to test the accuracy

    train_a = train[:len(train) / 2] # 1st half
    train_b = train[len(train) / 2:] # 2nd half

    test = list(DictReader(open("../data/spoilers/test.csv", 'r')))

    # Get features
    analyzer = Analyzer(flags.word, flags.page, flags.trope)

    feat = Featurizer(analyzer)

    labels = []
    for line in train_a:
        if not line[kTARGET_FIELD] in labels:
            labels.append(line[kTARGET_FIELD])

    x_train_a = feat.train_feature(x[kTEXT_FIELD] for x in train_a)
    x_train_b = feat.test_feature(x[kTEXT_FIELD] for x in train_b)
    x_test = feat.test_feature(x[kTEXT_FIELD] for x in test)

    y_train_a = array(list(labels.index(x[kTARGET_FIELD]) for x in train_a))
    y_train_b = array(list(labels.index(x[kTARGET_FIELD]) for x in train_b))

    print(len(train), len(y_train_a))
    print(set(y_train_a))

    # Train classifier
    lr = SGDClassifier(loss='log', penalty='l2', shuffle=True)
    lr.fit(x_train_a, y_train_a)

    feat.show_top10(lr, labels) # show the top 10 features used for classification

    # show the accuracy
    train_predictions_a = lr.predict(x_train_a)
    print("Training Accuracy A: %f" % accuracy_score(y_train_a, train_predictions_a))
    train_predictions_b = lr.predict(x_train_b)
    print("Training Accuracy B (testing simulation): %f" % accuracy_score(y_train_b, train_predictions_b))

    predictions = lr.predict(x_test)
    o = DictWriter(open("predictions.csv", 'w'), ["id", "cat"])
    o.writeheader()
    for ii, pp in zip([x['id'] for x in test], predictions):
        d = {'id': ii, 'cat': labels[pp]}
        o.writerow(d)
