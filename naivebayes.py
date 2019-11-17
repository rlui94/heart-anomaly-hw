"""Naive Bayes implementation"""
import csv


class Learner:
    """
    Contains information on the dataset, performs training, and classifies.
    """

    def __init__(self, num_features=0, num_classes=0):
        self.num_features = num_features
        self.num_classes = num_classes
        self.feature_class = []
        self.classes = []

    def read_csv(self, filename):
        """
        Reads data in from csv file
        :param filename: path to csv file as a string
        :return: list of data read from csv file
        """
        with open(filename) as file:
            reader = csv.reader(file, delimiter=',')
            dataset = list()
        return dataset

    def separate_by_class(self, dataset):
        """
        https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/
        Separates dataset into its given classes. In this case, there are only two classes: normal and abnormal.
        :param dataset: Data to be separated into classes
        :return: dictionary of lists where the key is the class and value is a list of vectors in that class.
        """
        classes = dict()
        for i in range(len(dataset)):
            vector = dataset[i]
            class_value = vector[0]
            if class_value not in classes:
                classes[class_value] = list()
            classes[class_value].append(vector)
        return classes

    def train(self, training_set):
        """
        Train learner via training_set
        :param training_set: List of training instances t has a class t.c and array of features t.f
        :return:feature_class: 2-dimensional array of counts feature_class[i,j] where left dimen is
            an instance classification (0 or 1) and right dimension is a feature number.
            Contents of each array entry is a count of the number of times
            that the given feature appears positive in the given class.
        :return:classes: Array classes[i] indexed by instance classification and gives the count of training instances with
            that class.
        """
        feature_class = [[0 for i in range(self.num_features)] for i in range(self.num_classes)]
        classes = [0, 0]
        for t in training_set:
            classes[t[0]] += 1
            j = 0
            for feature in t[1:]:
                if feature == 1:
                    feature_class[t[0]][j] += 1
                j += 1
        return feature_class, classes

    def classify(self):
        return 0
