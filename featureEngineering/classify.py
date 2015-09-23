#!/usr/bin/env python
#
# Starting code from:
# http://www.cs.colorado.edu/~jbg/teaching/CSCI_5622/04.py
# https://github.com/ezubaric/ml-hw/blob/master/feat_eng/classify.py
# 
# Modified By:
# Paul Boschert <paul@boschert.net>, <paul.boschert@colorado.edu>
# CSCI 5622 - Machine Learning: Feature Engineering (HW 3)
#


#from nltk.util import ngrams
from sklearn.metrics import accuracy_score
import argparse
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from sklearn.cross_validation import train_test_split
import re
import json
import urllib

from csv import DictReader, DictWriter
import csv

import numpy as np
from numpy import array

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier

kTEXT_FIELD   = 'sentence'
kTARGET_FIELD = 'spoiler'
kVERB_FIELD   = 'verb'
kPAGE_FIELD   = 'page'
kTROPE_FIELD  = 'trope'

def camelCaseConvert(value):
    '''
    Convert a string that is CamelCase to multiple words, also change to lower case
    For example CamelCase becomes camel case

    :param value The string to convert
    :return A lower case string with spaces separating the humps of the camel case word
    '''
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1 \2', value)
    return re.sub('([a-z0-9])([A-Z])', r'\1 \2', s1).lower()

def queryGenre(title):
    '''
    Query the genre from the open movie database

    :param title The title of the item to search.  Titles containing numbers like the year of production will be removed
    :return The genre matching the title if found, an empty string if not found
    '''
    genre = ""
    for yearInTitle in range(1888, 2015):
        if str(yearInTitle) in title:
            title = title.replace(str(yearInTitle), " ")

    # query the open movie database
    queryURL = "http://www.omdbapi.com/?t=" + title + "&tomatoes=true"
    response = urllib.urlopen(queryURL).read()
    movieInfo = json.loads(response)

    # if a genre is defined for this title, return the first one listed (presumably this is the highest priority)
    if 'Genre' in movieInfo:
        genre = movieInfo['Genre'].split()[0].replace(',', '')

    # return the genre found
	return genre

def queryYear(title):
    '''
    Query the year from the open movie database

    :param title The title of the item to search.  Titles containing numbers like the year of production will be removed
    :return The year(s) matching the title if found, an empty string if not found
    '''
    year = ""
    for yearInTitle in range(1888, 2015):
        if str(yearInTitle) in title:
            title = title.replace(str(yearInTitle), " ")

    # query the open movie database
    queryURL = "http://www.omdbapi.com/?t=" + title + "&tomatoes=true"
    response = urllib.urlopen(queryURL).read()
    movieInfo = json.loads(response)

    # if year(s) are defined for this title
    if 'Year' in movieInfo:
        year = movieInfo['Year']

    # remove unknown unicode dashes
    year = year.encode('ascii', 'replace')
    year = year.replace('?', '-')

    # return the year(s) found
    return year

def addGenres(dataset, field, X, cached_genres):
    for x, i in zip(dataset, range(len(dataset))):
        title = camelCaseConvert(x[field])

        # query the genre if it doesn't exist in the cached dictionary
        if title not in cached_genres:
            print("'%s' not found in cached_genres, querying..." % title)
            cached_genres[title] = queryGenre(title)
            print("   '%s' query resulted in genre: %s" % (title, cached_genres[title]))

        if title in cached_genres:
            genre = ''.join(("Genre", cached_genres[title]))
            X[i] = ' '.join((X[i], genre))

    return (X, cached_genres)


def addYears(dataset, field, X, cached_years):
    for x, i in zip(dataset, range(len(dataset))):
        title = camelCaseConvert(x[field])

        # query the year(s) if they don't exist in the cached dictionary
        if title not in cached_years:
            print("'%s' not found in cached_years, querying..." % title)
            cached_years[title] = queryYears(title)
            print("   '%s' query resulted in year: %s" % (title, cached_years[title]))

        if title in cached_years:
            year = ''.join(("Year", cached_years[title]))
            X[i] = ' '.join((X[i], year))

    return (X, cached_years)

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
                                          #strip_accents = "ascii",
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
    parser.add_argument('--word', default = False, action = "store_true",
                        help = "Use the words of the text as features")
    parser.add_argument('--page', default = False, action = "store_true",
                        help = "Use the page as a feature, this appears to be the movie/show that the quote is about")
    parser.add_argument('--trope', default = False, action = "store_true",
                        help = "Use the trope as a feature, this appears to be the author of the quote")
    parser.add_argument('--split', default = False, action = "store_true",
                        help = "Use the trope as a feature, this appears to be the movie/show that the quote is about")
    parser.add_argument('--genre', default = False, action = "store_true",
                        help = "Add the genre based on the page")
    parser.add_argument('--year', default = False, action = "store_true",
                        help = "Add the year based on the page")
    parser.add_argument('--decade', default = False, action = "store_true",
                        help = "Use the decade instead of the year based on the page")

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
        x_test = array(list(' '.join((x[kTEXT_FIELD], x[kTROPE_FIELD], x[kPAGE_FIELD])) for x in test))
    elif flags.trope:
        x_train_all = array(list(' '.join((x[kTEXT_FIELD], x[kTROPE_FIELD])) for x in train))
        x_test = array(list(' '.join((x[kTEXT_FIELD], x[kTROPE_FIELD])) for x in test))
    elif flags.page:
        x_train_all = array(list(' '.join((x[kTEXT_FIELD], x[kPAGE_FIELD])) for x in train))
        x_test = array(list(' '.join((x[kTEXT_FIELD], x[kPAGE_FIELD])) for x in test))
    else:
        x_train_all = array(list((x[kTEXT_FIELD] for x in train)))
        x_test = array(list((x[kTEXT_FIELD] for x in test)))

    y_train_all = array(list(labels.index(x[kTARGET_FIELD]) for x in train))

    if flags.genre:
        # read in the cached genres file
        cached_genres = {}
        try:
            reader = csv.reader(open("../data/spoilers/cached_genres.csv", 'r'))
            for row in reader:
                key, value = row
                cached_genres[key] = value;
        except:
            print("WARNING: No cached genre file found... go get some coffee")

        (x_train_all, cached_genres) = addGenres(train, kPAGE_FIELD, x_train_all, cached_genres)
        (x_test, cached_genres) = addGenres(test, kPAGE_FIELD, x_test, cached_genres)

        # write out the cached genre lookup dictionary
        o = DictWriter(open("../data/spoilers/cached_genres.csv", 'w'), ["page", "genre"])
        o.writeheader()
        for title, genre in zip(cached_genres.keys(), cached_genres.values()):
            d = {'page': title, 'genre': genre}
            o.writerow(d)

    if flags.year:
        # read in the cached years file
        cached_years = {}
        try:
            reader = csv.reader(open("../data/spoilers/cached_years.csv", 'r'))
            for row in reader:
                key, value = row
                cached_years[key] = value;
        except:
            print("WARNING: No cached years file found... go get some coffee")

        (x_train_all, cached_years) = addYears(train, kPAGE_FIELD, x_train_all, cached_years)
        (x_test, cached_years) = addYears(test, kPAGE_FIELD, x_test, cached_years)

        # write out the cached year lookup dictionary
        o = DictWriter(open("../data/spoilers/cached_years.csv", 'w'), ["page", "year"])
        o.writeheader()
        for title, year in zip(cached_years.keys(), cached_years.values()):
            d = {'page': title, 'year': year}
            o.writerow(d)

    # since we don't have y values for our testing set, get an approximation for our testing
    # set by splitting our training set in two and using the first part to classify and the
    # second part to validate the accuracy
    if flags.split:
        x_train, x_validate, y_train, y_validate = train_test_split(x_train_all, y_train_all, test_size = .25,
                                                                    random_state = 0)
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

    # train
    x_train = feat.train_feature(x_train)

    if flags.split:
        # test the validation data
        x_validate = feat.test_feature(x_validate)

    # test the actual test data
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

