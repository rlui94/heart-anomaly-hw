"""K-Nearest Neighbor implementation"""
import csv
import math
import random

FILEFLAG = 3
K = 5  # Use only ODD numbers since we have an even number of classes!


def file_switch(flag):
    switcher = {
        1: ('spect-orig.train.csv', 'spect-orig.test.csv', 'orig'),
        2: ('spect-itg.train.csv', 'spect-itg.test.csv', 'itg'),
        3: ('spect-resplit.train.csv', 'spect-resplit.test.csv', 'resplit'),
    }
    return switcher.get(flag, "Invalid flag")


class Learner:
    """
    Contains information on the dataset, performs training, and classifies.
    """

    def __init__(self):
        """
        Constructor
        """
        self.dataset = []

    def read_csv(self, filename):
        """
        Reads data for testing in from csv file
        :param filename: path to csv file as a string
        :return:
        """
        with open(filename) as file:
            reader = csv.reader(file, delimiter=',')
            dataset = list()
            for row in reader:
                vector = list()
                for x in row:
                    vector.append(int(x))
                dataset.append(vector)
        self.dataset = dataset

    def distance(self, v1, v2):
        """
        Calculates euclidean distance given two vectors. Assume first integer in each list is the classification.
        :param v1: vector 1 as list of integers
        :param v2: vector 2 as list of integers
        :return: distance as float
        """
        dist = 0.0
        for i in range(1, len(v1)):
            dist += (v2[i] - v1[i]) ** 2
        return math.sqrt(dist)

    def distance_h(self, v1, v2):
        """
        Calculates hamming distance given two vectors. Assume first integer in each list is the classification.
        :param v1: vector 1 as list of integers
        :param v2: vector 2 as list of integers
        :return: distance as integer
        """
        dist = 0
        for i in range(1, len(v1)):
            dist += v2[i] ^ v1[i]
        return dist

    def neighbors(self, instance, dataset):
        """
        Create a list of distances from the given instance to each vector in the dataset
        :param instance: instance vector as a list of int
        :param dataset: list of vectors
        :return: list of tuples (i, j) where i is the distance to the instance and j is the instance's class
        """
        distances = list()
        for vector in dataset:
            distances.append((self.distance_h(instance, vector), vector[0]))
        return distances

    def k_nearest(self, instance, dataset, k):
        """
        Finds the k nearest neighbors for the given instance vector in a dataset of vectors
        :param instance: instance vector as list of integers
        :param dataset: dataset of vectors as a list of vectors
        :param k: the number of instances to return
        :return: list of k tuples (i, j) where i is the distance to the instance and j is the instance's class
        """
        neighbors = self.neighbors(instance, dataset)
        neighbors.sort(key=lambda neighbor: neighbor[0])
        return neighbors[:k]

    def classify(self, instance, dataset, k):
        """
        Finds the k nearest neighbors for the given instance vector in a dataset of vectors,
        then classifies the instance based on the neighbors found.
        :param instance: instance vector as list of integers
        :param dataset: dataset of vectors as a list of vectors
        :param k: the number of instances to return
        :return: class of instance as an integer
        """
        neighbors = self.neighbors(instance, dataset)
        neighbors.sort(key=lambda neighbor: neighbor[0])
        classes = dict()
        for tup in neighbors[:k]:
            if tup[1] not in classes:
                classes[tup[1]] = 1
            else:
                classes[tup[1]] += 1
        return max(classes, key=classes.get)

    def classify_check(self, dataset, k):
        """
        Classify a set of instances using training data and return number of instances correctly classified.
        :param dataset: 2-dimensional array of integers
        :param k: Number of closest neighbors to examine
        :return: string showing accuracy, true negative rate, and true positive rate in the format
        (instances correct)/(total instances)(percent correct)
        """
        correct = 0
        class_instances_neg = 0
        class_instances_pos = 0
        true_neg = 0
        true_pos = 0
        for inst in dataset:
            inst_class = self.classify(inst, dataset, k)
            if inst[0] == 1:
                class_instances_pos += 1
                if inst_class == inst[0]:
                    correct += 1
                    true_pos += 1
            else:
                class_instances_neg += 1
                if inst_class == inst[0]:
                    correct += 1
                    true_neg += 1
        ret = ("%d/%d(%5.3f) %d/%d(%5.3f) %d/%d(%5.3f)\n" % (correct, len(dataset), correct/len(dataset), true_neg, class_instances_neg, true_neg/class_instances_neg, true_pos, class_instances_pos, true_pos/class_instances_pos))
        return ret

    def solve(self):
        """
        Main method call. Uses file_switch to choose data file and classifies the data.
        Shuffle the dataset to prevent ties.
        :return: string with name of data showing accuracy, true negative rate, and true positive rate in the format
        (instances correct)/(total instances)(percent correct)
        """
        data = file_switch(FILEFLAG)
        self.read_csv(data[1])
        random.shuffle(self.dataset)
        return data[2] + " " + self.classify_check(self.dataset, K)
