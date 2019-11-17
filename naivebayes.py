"""Naive Bayes implementation
    Followed tutorial at https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/"""
import csv

def read_csv(filename):
    """
    Reads data in from csv file
    :param filename: path to csv file as a string
    :return: list of data read from csv file
    """
    with open(filename) as file:
        reader = csv.reader(file, delimiter=',')
        dataset = list()
    return dataset

def separate_by_class(dataset):
    """
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