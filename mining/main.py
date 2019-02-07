from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from collections import defaultdict
from itertools import product

from gspan_mining import gSpan
from gspan_mining import GraphDatabase
from heapq import heappush, heappop

INF = float('inf')


class PatternGraphs:
    """
    This template class is used to define a task for the gSpan implementation.
    You should not modify this class but extend it to define new tasks
    """

    def __init__(self, database):
        # A list of subsets of graph identifiers.
        # Is used to specify different groups of graphs (classes and training/test sets).
        # The gid-subsets parameter in the pruning and store function will contain for each subset, all the occurrences
        # in which the examined pattern is present.
        self.gid_subsets = []

        self.database = database  # A graphdatabase instance: contains the data for the problem.

    def store(self, dfs_code, gid_subsets):
        """
        Code to be executed to store the pattern, if desired.
        The function will only be called for patterns that have not been pruned.
        In correlated pattern mining, we may prune based on confidence, but then check further conditions before storing.
        :param dfs_code: the dfs code of the pattern (as a string).
        :param gid_subsets: the cover (set of graph ids in which the pattern is present) for each subset in self.gid_subsets
        """
        print("Please implement the store function in a subclass for a specific mining task!")

    def prune(self, gid_subsets):
        """
        prune function: used by the gSpan algorithm to know if a pattern (and its children in the search tree)
        should be pruned.
        :param gid_subsets: A list of the cover of the pattern for each subset.
        :return: true if the pattern should be pruned, false otherwise.
        """
        print("Please implement the prune function in a subclass for a specific mining task!")


class TopDict:
    def __init__(self, maxsize):
        self.maxsize = maxsize
        self._td = {}
        self._list = []
        self.length = 0

    def competitive_push(self, key, value):

        # Put the key, value pair with the dictionary
        # Note : if the key already exists, concatenate the values within a list
        # Ex :
        #   x = dict(2: ['A'], 3: ['C'])
        #   x.competitive_push(2, 'B')
        #   x = dict(2: ['A', 'B'], 3: ['C'])
        if key in self._td:
            self._td[key].append(value)
        else:
            heappush(self._list, key)
            self._td[key] = [value]

        self.length += 1

        # If the size of the dictionary (in the keys) exceeds maxsize, dropout the element with the lowest key
        # to maintain a number k of top elements
        if len(self._td) > self.maxsize:
            key = heappop(self._list)
            self.length -= len(self._td[key])
            self._td.pop(key)

    def get_min(self):
        # Get the smallest key
        return self._list[0]

    def __len__(self):
        return self.length

    def __repr__(self):
        return repr(self._td)

    def __iter__(self):
        # Iterate the elements in decreasing value of their keys (greatest key first)
        for key in sorted(self._list, reverse=True):
            #for value in sorted(self._td[key]): # For phase 3
            for value in self._td[key]:
                yield key, value


class FrequentGraphs(PatternGraphs):

    def __init__(self, database, subsets, k, minsup):
        super().__init__(database)
        self.patterns = TopDict(maxsize=k)
        self.gid_subsets = subsets
        self.k = k
        self.minsup = minsup

    # Stores any pattern found that has not been pruned
    def store(self, dfs_code, gid_subsets):
        p = len(gid_subsets[0])
        n = len(gid_subsets[2])
        score = ((p/(p+n)), p+n)
        self.patterns.competitive_push(score, (dfs_code, gid_subsets))

    # Prunes any pattern that is not frequent in the positive class
    def prune(self, gid_subsets):
        p = len(gid_subsets[0])
        n = len(gid_subsets[2])

        return p+n < self.minsup

    # creates a column for a feature matrix
    def create_fm_col(self, all_gids, subset_gids):
        subset_gids = set(subset_gids)
        bools = []
        for i, val in enumerate(all_gids):
            if val in subset_gids:
                bools.append(1)
            else:
                bools.append(0)
        return bools

    # return a feature matrix for each subset of examples, in which the columns correspond to patterns
    # and the rows to examples in the subset.
    def get_feature_matrices(self):
        matrices = [[] for _ in self.gid_subsets]
        for score, (pattern, gid_subsets) in self.patterns:
            for i, gid_subset in enumerate(gid_subsets):
                matrices[i].append(self.create_fm_col(self.gid_subsets[i], gid_subset))
        return [numpy.array(matrix).transpose() for matrix in matrices]

        # matrices = [[] for _ in self.gid_subsets]
        # for score, (pattern, gid_subsets) in self.patterns:
        #     for i, gid_subset in enumerate(gid_subsets):
        #         matrices[i].append(self.create_fm_col(self.gid_subsets[i], gid_subset))
        # return [numpy.array(matrix).transpose() for matrix in matrices]


class FrequentGraphs3(PatternGraphs):

    def __init__(self, database, subsets, k, minsup):
        super().__init__(database)
        self.patterns = TopDict(maxsize=k)
        self.gid_subsets = subsets
        self.k = k
        self.minsup = minsup

    # Stores any pattern found that has not been pruned
    def store(self, dfs_code, gid_subsets):
        p = len(gid_subsets[0])
        n = len(gid_subsets[2])
        value_to_predict = 1 if p/(p+n) >= n/(p+n) else -1
        score = (max(p/(p+n), n/(p+n)), p+n)
        self.patterns.competitive_push(score, (dfs_code, gid_subsets, value_to_predict))

    # Prunes any pattern that is not frequent in the positive class
    def prune(self, gid_subsets):
        p = len(gid_subsets[0])
        n = len(gid_subsets[2])

        return p+n < self.minsup

    # creates a column for a feature matrix
    def create_fm_col(self, all_gids, subset_gids):
        subset_gids = set(subset_gids)
        bools = []
        for i, val in enumerate(all_gids):
            if val in subset_gids:
                bools.append(1)
            else:
                bools.append(0)
        return bools

    # return a feature matrix for each subset of examples, in which the columns correspond to patterns
    # and the rows to examples in the subset.
    def get_feature_matrices(self):
        matrices = [[] for _ in self.gid_subsets]
        for score, (pattern, gid_subsets, value_to_predict) in self.patterns:
            for i, gid_subset in enumerate(gid_subsets):
                matrices[i].append(self.create_fm_col(self.gid_subsets[i], gid_subset))
            break
        return [numpy.array(matrix).transpose() for matrix in matrices]

        # matrices = [[] for _ in self.gid_subsets]
        # for score, (pattern, gid_subsets) in self.patterns:
        #     for i, gid_subset in enumerate(gid_subsets):
        #         matrices[i].append(self.create_fm_col(self.gid_subsets[i], gid_subset))
        # return [numpy.array(matrix).transpose() for matrix in matrices]


def train_and_evaluate(k, minsup, database, subsets):
    task = FrequentGraphs(database, subsets, k, minsup)  # Creating task

    gSpan(task).run()  # Running gSpan

    # Creating feature matrices for training and testing:
    features = task.get_feature_matrices()
    train_fm = numpy.concatenate((features[0], features[2]))  # Training feature matrix
    train_labels = numpy.concatenate((numpy.full(len(features[0]), 1, dtype=int), numpy.full(len(features[2]), -1, dtype=int)))  # Training labels
    test_fm = numpy.concatenate((features[1], features[3]))  # Testing feature matrix
    test_labels = numpy.concatenate((numpy.full(len(features[1]), 1, dtype=int), numpy.full(len(features[3]), -1, dtype=int)))  # Testing labels

    classifier = tree.DecisionTreeClassifier()  # Creating model object
    classifier.fit(train_fm, train_labels)  # Training model

    predicted = classifier.predict(test_fm)  # Using model to predict labels of testing data

    accuracy = metrics.accuracy_score(test_labels, predicted)  # Computing accuracy:

    # Printing frequent patterns along with their positive support:
    for score, (pattern, gid_subsets) in task.patterns:
        print('{} {} {}'.format(pattern, *score))
    # printing classification results:
    print(predicted)
    print('accuracy: {}'.format(accuracy))
    print()  # Blank line to indicate end of fold.


def train_and_evaluate3(k, minsup, database, subsets, curr_fold, nfolds):

    set_pattern = set()
    current_subsets = subsets.copy()

    predicted = dict()
    for e in subsets[1] + subsets[3]:
        predicted[e] = 0

    for i in range(k):
        task = FrequentGraphs3(database, current_subsets, 1, minsup)  # Creating task
        gSpan(task).run()  # Running gSpan

        for score, (pattern, gid_subsets, value_to_predict) in task.patterns:
            print('{} {} {}'.format(pattern, *score))
            set_pattern.add(pattern)
            break

        features = task.get_feature_matrices()
        train_fm = numpy.concatenate((features[0], features[2]))  # Training feature matrix
        train_labels = numpy.concatenate((numpy.full(len(features[0]), 1, dtype=int), numpy.full(len(features[2]), -1, dtype=int)))  # Training labels
        test_fm = numpy.concatenate((features[1], features[3]))  # Testing feature matrix
        if i == 0:
            test_labels = numpy.concatenate((numpy.full(len(features[1]), 1, dtype=int), numpy.full(len(features[3]), -1, dtype=int)))  # Testing labels

        test_subsets = np.concatenate((current_subsets[1], current_subsets[3]))
        for j in range(len(test_fm)):
            if test_fm[j] == 1:
                #print('pattern is in transaction : {}'.format(test_subsets[i]))
                predicted[test_subsets[j]] = value_to_predict

        if i == (k-1):
            val = 1 if train_labels.tolist().count(1) >= train_labels.tolist().count(-1) else -1

            for m in predicted:
                if predicted[m] == 0:
                    predicted[m] = val

        else:
            if len(features[0]) != 0:
                to_delete_pos_train = np.where(features[0].all(axis=1))[0]
                current_subsets[0] = np.array([x for i, x in enumerate(current_subsets[0]) if i not in to_delete_pos_train])
            if len(features[1]) != 0:
                to_delete_pos_test = np.where(features[1].all(axis=1))[0]
                current_subsets[1] = np.array([x for i, x in enumerate(current_subsets[1]) if i not in to_delete_pos_test])
            if len(features[2]) != 0:
                to_delete_neg_train = np.where(features[2].all(axis=1))[0]
                current_subsets[2] = np.array([x for i, x in enumerate(current_subsets[2]) if i not in to_delete_neg_train])
            if len(features[3]) != 0:
                to_delete_neg_test = np.where(features[3].all(axis=1))[0]
                current_subsets[3] = np.array([x for i, x in enumerate(current_subsets[3]) if i not in to_delete_neg_test])

    predicted = (list(predicted.values()))

    print(predicted)

    print('accuracy: {}'.format(metrics.accuracy_score(test_labels, predicted)))

    if curr_fold != nfolds-1:
        print()  # Blank line to indicate end of fold.


def phase1():
    args = sys.argv
    database_file_name_pos = args[1]  # First parameter: path to positive class file
    database_file_name_neg = args[2]  # Second parameter: path to negative class file
    k = int(args[3]) # Third parameter: for the top k
    minsup = int(args[4]) # Fourth parameter: minimum support

    if not os.path.exists(database_file_name_pos):
        print('{} does not exist.'.format(database_file_name_pos))
        sys.exit()
    if not os.path.exists(database_file_name_neg):
        print('{} does not exist.'.format(database_file_name_neg))
        sys.exit()

    graph_database = GraphDatabase()  # Graph database object
    pos_ids = graph_database.read_graphs(database_file_name_pos)  # Reading positive graphs, adding them to database and getting ids
    neg_ids = graph_database.read_graphs(database_file_name_neg)  # Reading negative graphs, adding them to database and getting ids

    subsets = [pos_ids, neg_ids]  # The ids for the positive and negative labelled graphs in the database
    task = FrequentGraphs(graph_database, subsets, k, minsup)  # Creating task

    gSpan(task).run()  # Running gSpan

    # Printing frequent patterns along with their positive support:
    for key, value in task.patterns:
        print('{} {} {}'.format(value[0], *key))


def phase2():
    """
    Runs gSpan with the specified positive and negative graphs; finds all frequent subgraphs in the training subset of
    the positive class with a minimum support of minsup.
    Uses the patterns found to train a naive bayesian classifier using Scikit-learn and evaluates its performances on
    the test set.
    Performs a k-fold cross-validation.
    """

    args = sys.argv
    database_file_name_pos = args[1]  # First parameter: path to positive class file
    database_file_name_neg = args[2]  # Second parameter: path to negative class file
    k = int(args[3])
    minsup = int(args[4])  # Third parameter: minimum support (note: this parameter will be k in case of top-k mining)
    nfolds = int(args[5])  # Fourth parameter: number of folds to use in the k-fold cross-validation.

    if not os.path.exists(database_file_name_pos):
        print('{} does not exist.'.format(database_file_name_pos))
        sys.exit()
    if not os.path.exists(database_file_name_neg):
        print('{} does not exist.'.format(database_file_name_neg))
        sys.exit()

    graph_database = GraphDatabase()  # Graph database object
    pos_ids = graph_database.read_graphs(database_file_name_pos)  # Reading positive graphs, adding them to database and getting ids
    neg_ids = graph_database.read_graphs(database_file_name_neg)  # Reading negative graphs, adding them to database and getting ids

    # If less than two folds: using the same set as training and test set (note this is not an accurate way to evaluate the performances!)
    if nfolds < 2:
        subsets = [
            pos_ids,  # Positive training set
            pos_ids,  # Positive test set
            neg_ids,  # Negative training set
            neg_ids  # Negative test set
        ]
        # Printing fold number:
        print('fold {}'.format(1))
        train_and_evaluate(k, minsup, graph_database, subsets)

    # Otherwise: performs k-fold cross-validation:
    else:
        pos_fold_size = len(pos_ids) // nfolds
        neg_fold_size = len(neg_ids) // nfolds
        for i in range(nfolds):
            # Use fold as test set, the others as training set for each class;
            # identify all the subsets to be maintained by the graph mining algorithm.
            subsets = [
                numpy.concatenate((pos_ids[:i * pos_fold_size], pos_ids[(i + 1) * pos_fold_size:])),  # Positive training set
                pos_ids[i * pos_fold_size:(i + 1) * pos_fold_size],  # Positive test set
                numpy.concatenate((neg_ids[:i * neg_fold_size], neg_ids[(i + 1) * neg_fold_size:])),  # Negative training set
                neg_ids[i * neg_fold_size:(i + 1) * neg_fold_size],  # Negative test set
            ]
            # Printing fold number:
            print('fold {}'.format(i+1))
            train_and_evaluate(k, minsup, graph_database, subsets)


def phase3():
    """
    Runs gSpan with the specified positive and negative graphs; finds all frequent subgraphs in the training subset of
    the positive class with a minimum support of minsup.
    Uses the patterns found to train a naive bayesian classifier using Scikit-learn and evaluates its performances on
    the test set.
    Performs a k-fold cross-validation.
    """

    args = sys.argv
    database_file_name_pos = args[1]  # First parameter: path to positive class file
    database_file_name_neg = args[2]  # Second parameter: path to negative class file
    k = int(args[3])
    minsup = int(args[4])  # Third parameter: minimum support (note: this parameter will be k in case of top-k mining)
    nfolds = int(args[5])  # Fourth parameter: number of folds to use in the k-fold cross-validation.

    if not os.path.exists(database_file_name_pos):
        print('{} does not exist.'.format(database_file_name_pos))
        sys.exit()
    if not os.path.exists(database_file_name_neg):
        print('{} does not exist.'.format(database_file_name_neg))
        sys.exit()

    graph_database = GraphDatabase()  # Graph database object
    pos_ids = graph_database.read_graphs(database_file_name_pos)  # Reading positive graphs, adding them to database and getting ids
    neg_ids = graph_database.read_graphs(database_file_name_neg)  # Reading negative graphs, adding them to database and getting ids

    # If less than two folds: using the same set as training and test set (note this is not an accurate way to evaluate the performances!)
    if nfolds < 2:
        subsets = [
            pos_ids,  # Positive training set
            pos_ids,  # Positive test set
            neg_ids,  # Negative training set
            neg_ids  # Negative test set
        ]
        # Printing fold number:
        print('fold {}'.format(1))
        train_and_evaluate3(k, minsup, graph_database, subsets)

    # Otherwise: performs k-fold cross-validation:
    else:
        pos_fold_size = len(pos_ids) // nfolds
        neg_fold_size = len(neg_ids) // nfolds
        for i in range(nfolds):
            # Use fold as test set, the others as training set for each class;
            # identify all the subsets to be maintained by the graph mining algorithm.
            subsets = [
                numpy.concatenate((pos_ids[:i * pos_fold_size], pos_ids[(i + 1) * pos_fold_size:])),  # Positive training set
                pos_ids[i * pos_fold_size:(i + 1) * pos_fold_size],  # Positive test set
                numpy.concatenate((neg_ids[:i * neg_fold_size], neg_ids[(i + 1) * neg_fold_size:])),  # Negative training set
                neg_ids[i * neg_fold_size:(i + 1) * neg_fold_size],  # Negative test set
            ]
            # Printing fold number:
            print('fold {}'.format(i+1))
            train_and_evaluate3(k, minsup, graph_database, subsets, i, nfolds)


def phase4():
    """
    Runs gSpan with the specified positive and negative graphs; finds all frequent subgraphs in the training subset of
    the positive class with a minimum support of minsup.
    Uses the patterns found to train a naive bayesian classifier using Scikit-learn and evaluates its performances on
    the test set.
    Performs a k-fold cross-validation.
    """

    args = sys.argv
    database_file_name_pos = args[1]  # First parameter: path to positive class file
    database_file_name_neg = args[2]  # Second parameter: path to negative class file
    nfolds = int(args[3])  # Third parameter: number of folds to use in the k-fold cross-validation.
    file_to_write = str(args[4])
    model = int(args[5])

    graph_database = GraphDatabase()  # Graph database object
    pos_ids = graph_database.read_graphs(database_file_name_pos)  # Reading positive graphs, adding them to database and getting ids
    neg_ids = graph_database.read_graphs(database_file_name_neg)  # Reading negative graphs, adding them to database and getting ids

    pos_fold_size = len(pos_ids) // nfolds
    neg_fold_size = len(neg_ids) // nfolds

    f = open(file_to_write + "_" + str(model) + ".txt", "w")
    
    n_transactions = len(pos_ids) + len(neg_ids)
    k_list = np.linspace(1, n_transactions/2, 10, dtype=np.int)
    minsup_list = np.linspace(2, n_transactions/2, 10, dtype=np.int)

    accuracy_list = []

    for k in k_list:
        print('k = {}'.format(k))
        for minsup in minsup_list:
            acc_list = list()
            for i in range(nfolds):
                # Use fold as test set, the others as training set for each class;
                # identify all the subsets to be maintained by the graph mining algorithm.
                subsets = [
                    numpy.concatenate((pos_ids[:i * pos_fold_size], pos_ids[(i + 1) * pos_fold_size:])),
                    # Positive training set
                    pos_ids[i * pos_fold_size:(i + 1) * pos_fold_size],  # Positive test set
                    numpy.concatenate((neg_ids[:i * neg_fold_size], neg_ids[(i + 1) * neg_fold_size:])),
                    # Negative training set
                    neg_ids[i * neg_fold_size:(i + 1) * neg_fold_size],  # Negative test set
                ]
                # Printing fold number:
                #print('fold {}'.format(i + 1))
                acc = train_and_evaluate4(graph_database, subsets, k, minsup, model)
                acc_list.append(acc)
            accuracy = sum(acc_list)/len(acc_list)
            accuracy_list.append(accuracy)

            f.write('{} {} {}\n'.format(k, minsup, accuracy))

    accuracy_list.sort()
    print(accuracy_list[-1])


def train_and_evaluate4(database, subsets, k, minsup, model):

        task = FrequentGraphs(database, subsets, k, minsup)  # Creating task
        gSpan(task).run()  # Running gSpan

        # Creating feature matrices for training and testing:
        features = task.get_feature_matrices()
        train_fm = numpy.concatenate((features[0], features[2]))  # Training feature matrix
        train_labels = numpy.concatenate((numpy.full(len(features[0]), 1, dtype=int), numpy.full(len(features[2]), -1, dtype=int)))  # Training labels
        test_fm = numpy.concatenate((features[1], features[3]))  # Testing feature matrix
        test_labels = numpy.concatenate((numpy.full(len(features[1]), 1, dtype=int), numpy.full(len(features[3]), -1, dtype=int)))  # Testing labels

        # model == LogisticRegression
        if model == 1:
            classifier = LogisticRegression(random_state=1, solver='lbfgs')  # Creating model object
        # model == DecisionTree
        elif model == 2:
            classifier = tree.DecisionTreeClassifier(random_state=1)  # Creating model object
        # model == KNeighbors
        else:
            classifier = KNeighborsClassifier(random_state=1)  # Creating model object

        classifier.fit(train_fm, train_labels)  # Training model

        predicted = classifier.predict(test_fm)  # Using model to predict labels of testing data

        return metrics.accuracy_score(test_labels, predicted)  # Computing accuracy:


if __name__ == '__main__':
    #phase1()
    #phase2()
    #phase3()
    phase4()