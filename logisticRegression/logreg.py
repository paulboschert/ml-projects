#!/usr/bin/env python
#
# Code from: https://github.com/ezubaric/ml-hw/
#
# Modified By
# Paul Boschert <paul@boschert.net>, <paul.boschert@colorado.edu>
# CSCI 5622 - Machine Learning: Homework 2
# 


import random
from numpy import zeros, sign
from math import exp, log
from collections import defaultdict
import numpy as np

import argparse

kSEED = 1701
kBIAS = "BIAS_CONSTANT"

random.seed(kSEED)


def sigmoid(score, threshold=20.0):
    """
    Prevent overflow of exp by capping activation at 20.

    :param score: A real valued number to convert into a number between 0 and 1
    """

    if abs(score) > threshold:
        score = threshold * sign(score)

    activation = exp(score)
    return activation / (1.0 + activation)


class Example:
    """
    Class to represent a logistic regression example
    """
    def __init__(self, label, words, vocab, df):
        """
        Create a new example

        :param label: The label (0 / 1) of the example
        :param words: The words in a list of "word:count" format
        :param vocab: The vocabulary to use as features (list)
        """
        self.nonzero = {}
        self.y = label
        self.x = zeros(len(vocab))
        for word, count in [x.split(":") for x in words]:
            if word in vocab:
                assert word != kBIAS, "Bias can't actually appear in document"
                self.x[vocab.index(word)] += float(count)
                self.nonzero[vocab.index(word)] = word
        self.x[0] = 1

class LogReg:
    def __init__(self, num_features, mu, step=lambda x: 0.05):
        """
        Create a logistic regression classifier

        :param num_features: The number of features (including bias)
        :param mu: Regularization parameter
        :param step: A function that takes the iteration as an argument (the default is a constant value)
        """
        
        self.beta = zeros(num_features)
        self.mu = mu
        self.step = step
        self.last_update = defaultdict(int)

        assert self.mu >= 0, "Regularization parameter must be non-negative"

    def progress(self, examples):
        """
        Given a set of examples, compute the probability and accuracy

        :param examples: The dataset to score
        :return: A tuple of (log probability, accuracy)
        """

        logprob = 0.0
        num_right = 0
        for ii in examples:
            p = sigmoid(self.beta.dot(ii.x))
            if ii.y == 1:
                logprob += log(p)
            else:
                logprob += log(1.0 - p)

            # Get accuracy
            if abs(ii.y - p) < 0.5:
                num_right += 1

        return logprob, float(num_right) / float(len(examples))

    def sg_update(self, train_example, iteration, use_tfidf=False):
        """
        Compute a stochastic gradient update to improve the log likelihood.

        :param train_example: The example to take the gradient with respect to
        :param iteration: The current iteration (an integer)
        :param use_tfidf: A boolean to switch between the raw data and the tfidf representation
        :return: Return the new value of the regression coefficients
        """

        # allow easier access to these variables
        y = train_example.y
        mu = self.mu
        ada = self.step(iteration)

        # calculate the pi value once
        expVal = np.sum(np.multiply(self.beta, train_example.x))
        pi_I = exp(expVal) / (1 + exp(expVal))

        for (j, i) in zip(range(len(self.beta)), range(len(train_example.x))):
            x_i = train_example.x[i]
            
            # only update the beta value if there exists relevant data, if there isn't any, increment the time since the last update
            # This case is known as the zero-dimension
            if int(x_i) is 0:
                if i in self.last_update.keys():
                    self.last_update[i] += 1
                else:  # if there does not yet exist an entry in the dictionary, create one
                    self.last_update[i] = 1
            else:
                # Use a power of 1 unless we haven't updated this particular feature in the last iteration
                m_i = 1

                # if we haven't updated the feature in the previous iteration, get the last time we did and add 1
                if i in self.last_update.keys():
                    m_i = self.last_update[i] + 1

                self.beta[j] = (self.beta[j] + ada * (y - pi_I) * x_i) * np.power((1 - 2 * ada * mu), m_i)

                # remove the key if it exists since we just updated it
                if i in self.last_update.keys():
                    del(self.last_update[i])

        return self.beta


def read_dataset(positive, negative, vocab, test_proportion=.1):
    """
    Reads in a text dataset with a given vocabulary

    :param positive: Positive examples
    :param negative: Negative examples
    :param vocab: A list of vocabulary words
    :param test_proprotion: How much of the data should be reserved for test
    """
    df = [float(x.split("\t")[1]) for x in open(vocab, 'r') if '\t' in x]
    vocab = [x.split("\t")[0] for x in open(vocab, 'r') if '\t' in x]
    assert vocab[0] == kBIAS, \
        "First vocab word must be bias term (was %s)" % vocab[0]

    train = []
    test = []
    for label, input in [(1, positive), (0, negative)]:
        for line in open(input):
            ex = Example(label, line.split(), vocab, df)
            if random.random() <= test_proportion:
                test.append(ex)
            else:
                train.append(ex)

    # Shuffle the data so that we don't have order effects
    random.shuffle(train)
    random.shuffle(test)

    return train, test, vocab

def step_update(iteration):
    # from http://research.microsoft.com/pubs/192769/tricks-2012.pdf
    #y_0 = .5
    #return y_0 / (1.0 + y_0 / iteration);
    return 1 / iteration;

def printMinMaxPredictors(betas, vocab, num = 5):
    # get the indices that would sort the array, beta
    sortIndices = np.argsort(betas)

    # we want to 
    for i in range(1, len(sortIndices)):
        indexInBeta = sortIndices[i]
        if(betas[indexInBeta - 1] < 0 and betas[indexInBeta] > 0):
            print("Worst Indicator Num Hockey, Word, Beta")
            for j in list(reversed(range(num))):
                index = sortIndices[i - j - 1]
                print("%d, %s, %.6g" % (j, vocab[index], betas[index]))
            print("Worst Indicator Num Baseball, Word, Beta")
            for j in range(num):
                index = sortIndices[i + j]
                print("%d, %s, %.6g" % (j, vocab[index], betas[index]))
            break

    # print out the top 'num' words, these are the most negative beta values
    print("Best Indicator Num Hockey, Word, Beta")
    for i in range(num):
        index = sortIndices[i]
        print("%d, %s, %.6f" % (i, vocab[index], betas[index]))

    # print out the last 'num' words, these are the most positive beta values
    print("Best Indicator Num Baseball, Word, Beta")
    for i in range(num):
        index = sortIndices[len(sortIndices) - 1 - i]
        print("%d, %s, %.6f" % (i, vocab[index], betas[index]))

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--mu", help="Weight of L2 regression",
                           type=float, default=0.0, required=False)
    argparser.add_argument("--step", help="Initial SG step size",
                           type=float, default=0.1, required=False)
    argparser.add_argument("--positive", help="Positive class",
                           type=str, default="../data/hockey_baseball/positive", required=False)
    argparser.add_argument("--negative", help="Negative class",
                           type=str, default="../data/hockey_baseball/negative", required=False)
    argparser.add_argument("--vocab", help="Vocabulary that can be features",
                           type=str, default="../data/hockey_baseball/vocab", required=False)
    argparser.add_argument("--passes", help="Number of passes through train",
                           type=int, default=1, required=False)

    args = argparser.parse_args()
    train, test, vocab = read_dataset(args.positive, args.negative, args.vocab)

    print("Read in %i train and %i test" % (len(train), len(test)))

    # Initialize model
    lr = LogReg(len(vocab), args.mu, lambda x: args.step)

    # Iterations
    update_number = 0
    for pp in xrange(args.passes):
        for ii in train:
            update_number += 1
            betas = lr.sg_update(ii, update_number)

            if update_number % 5 == 1:
                # log probability, accuracy for the training data set
                train_lp, train_acc = lr.progress(train)

                # log probability, accuracy for the test data set
                ho_lp, ho_acc = lr.progress(test)
                print("Update %i\tTP %f\tHP %f\tTA %f\tHA %f" %
                      (update_number, train_lp, ho_lp, train_acc, ho_acc))

    # print out the min/max predictors
    #printMinMaxPredictors(betas, vocab, 5)

