"""Naive Bayes implementation"""
import csv
import math


class Learner:
    """
    Contains information on the dataset, performs training, and classifies.
    """

    def __init__(self, num_features=0, num_classes=0):
        """
        Constructor
        feature_class: 2-dimensional array of counts feature_class[i,j] where left dimen is
            an instance classification (0 or 1) and right dimension is a feature number.
            Contents of each array entry is a count of the number of times
            that the given feature appears positive in the given class.
        classes: Array classes[i] indexed by instance classification and gives the count of training instances with
            that class."""
        self.num_features = num_features
        self.num_classes = num_classes
        self.feature_class = []
        self.classes = []

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
        """
        self.feature_class = [[0 for i in range(self.num_features)] for i in range(self.num_classes)]
        self.classes = [0, 0]
        for t in training_set:
            self.classes[t[0]] += 1
            j = 0
            for feature in t[1:]:
                if feature == 1:
                    self.feature_class[t[0]][j] += 1
                j += 1

    def classify(self, instance):
        """
        Classify a particular instance using training data
        :param instance: classification instance to be classified as a list of integers
        :return: 1 if class 1, 0 if class 0
        """
        likelihood = [0 for i in range(self.num_classes)]
        instance_features = instance[1:]
        for i in range(0, self.num_classes):
            likelihood[i] = math.log(self.classes[i] + 0.5) - math.log(self.classes[0] + self.classes[1] + 0.5)
            j = 1
            for feature in range(1, self.num_features):
                s = self.feature_class[i][j]
                if instance_features[j] == 0:
                    s = self.classes[i] - s
                likelihood[i] = likelihood[i] + math.log(s + 0.5) - math.log(self.classes[i] + 0.5)
                j += 1
        if likelihood[0] > likelihood[1]:
            return 1
        return 0

    def classify_check(self, class_instances):
        """
        Classify a set of instances using training data and return number of instances correctly classified.
        :param class_instances: 2-dimensional array of integers
        :return: integer of instances correctly classified
        """
        correct = 0
        class_instances_neg = 0
        class_instances_pos = 0
        true_neg = 0
        true_pos = 0
        for inst in class_instances:
            inst_class = self.classify(inst)
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
        print("%d/%d(%d) %d/%d(%d) %d/%d(%d)\n" % (correct, len(class_instances), correct/len(class_instances), true_neg, class_instances_neg, true_neg/class_instances_neg, true_pos, class_instances_pos, true_pos/class_instances_pos))