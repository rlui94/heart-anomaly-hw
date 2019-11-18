"""K-Nearest Neighbor implementation"""
import csv
import math


class Learner:
    """
    Contains information on the dataset, performs training, and classifies.
    """

    def __init__(self, num_features=0, num_classes=0):
        """
        Constructor
        """
        self.num_features = num_features
        self.num_classes = num_classes

    def read_csv_train(self, filename):
        """
        Reads data for training in from csv file
        :param filename: path to csv file as a string
        :return: list of data read from csv file
        """
        with open(filename) as file:
            reader = csv.reader(file, delimiter=',')
            dataset = list()
            for row in reader:
                vector = list()
                for x in row:
                    vector.append(int(x))
                dataset.append(vector)
        self.num_features = len(dataset[0]) - 1
        classes = list()
        for i in dataset:
            if i[0] not in classes:
                classes.append(i[0])
        self.num_classes = len(classes)
        return dataset

    def read_csv_test(self, filename):
        """
        Reads data for testing in from csv file
        :param filename: path to csv file as a string
        :return: list of data read from csv file
        """
        with open(filename) as file:
            reader = csv.reader(file, delimiter=',')
            dataset = list()
            for row in reader:
                vector = list()
                for x in row:
                    vector.append(int(x))
                dataset.append(vector)
        return dataset

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

    def neighbors(self, instance, dataset):
        distances = list()
        for vector in dataset:
            distances.append(self.distance(instance, vector))