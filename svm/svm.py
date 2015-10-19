#!/usr/bin/env python
#
# Starting code from:
# https://github.com/ezubaric/ml-hw/blob/master/svm/svm.py
#
# Modified By:
# Paul Boschert <paul@boschert.net>, <paul.boschert@colorado.edu>
# CSCI 5622 - Machine Learning: Support Vector Machines (HW 5)

from numpy import array, zeros

kINSP = array([(1, 8, +1),
               (7, 2, -1),
               (6, -1, -1),
               (-5, 0, +1),
               (-5, 1, -1),
               (-5, 2, +1),
               (6, 3, +1),
               (6, 1, -1),
               (5, 2, -1)])

kSEP = array([
              (-2, 2, +1),    # 0 - A
              (0, 4, +1),     # 1 - B
              (2, 1, +1),     # 2 - C
              (-2, -3, -1),   # 3 - D
              (0, -1, -1),    # 4 - E
              (2, -3, -1),    # 5 - F
              ])

class Numbers:
    """
    Class to store MNIST data
    """

    def __init__(self, location):
        import cPickle, gzip

        # Load the dataset
        f = gzip.open(location, 'rb')
        train_set, valid_set, test_set = cPickle.load(f)

        self.train_x, self.train_y = train_set
        self.test_x, self.test_y = valid_set
        f.close()

def weight_vector(x, y, alpha):
    """
    Given a vector of alphas, compute the primal weight vector.
    """

    w = zeros(len(x[0]))

    # w = alpha * y * x
    for i, (alphav, yv) in enumerate(zip(alpha, y)):
        for j, xv in enumerate(x[i]):
            w[j] += alphav * yv * xv

    return w


def find_support(x, y, w, b, tolerance = 0.001):
    """
    Given a primal support vector, return the indices for all of the support
    vectors
    """

    support = set()

    # find the positive support vectors: where w dot x + b <= 1
    # and find where negative support vectors: where w dot x + b <= -1
    # use the given tolerance because we're dealing with floating point numbers and python likes to
    # store what seems an infinite amount of precision
    for i, xv in enumerate(x):
        if abs(w.dot(xv) + b) <= 1 + abs(tolerance) or abs(w.dot(xv) + b) <= -1 + abs(tolerance):
            support.add(i)

    return support


def find_slack(x, y, w, b):
    """
    Given a primal support vector instance, return the indices for all of the
    slack vectors
    """

    slack = set()

    # find the values that are essentially mis-classified, 1 y-values have negative w dot
    # x + b values and -1 y-values have positive w dot x + b values
    for i, (xv, yv) in enumerate(zip(x, y)):
        if yv == 1:
            if w.dot(xv) + b <= 0:
                slack.add(i)
        elif yv == -1:
            if w.dot(xv) + b >= 0:
                slack.add(i)

    return slack

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SVM classifier options')
    parser.add_argument('--limit', type=int, default=-1, help="Restrict training to this many examples")
    args = parser.parse_args()

    data = Numbers("../data/mnist.pkl.gz")

'''
    if args.limit > 0:
        print("Data limit: %i" % args.limit)
        knn = Knearest(data.train_x[:args.limit], data.train_y[:args.limit], args.k)
    else:
        knn = Knearest(data.train_x, data.train_y, args.k)
    print("Done loading data")

    confusion = knn.confusion_matrix(data.test_x, data.test_y)

    print("\t" + "\t".join(str(x) for x in xrange(10)))
    print("".join(["-"] * 90))

    for ii in xrange(10):
        print("%i:\t" % ii + "\t".join(str(confusion[ii].get(x, 0)) for x in xrange(10)))

    print("Accuracy: %f" % knn.acccuracy(confusion))
'''
