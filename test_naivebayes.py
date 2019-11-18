import unittest
from naivebayes import Learner

default_dataset = [
    # 23 numbers, 22 features, 1st is the class (1 of 2)
    [1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
]

default_dict = {
    0: [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
    1: [[1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0],
        [1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1]]
}

default_feature_class = [
    # 0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    # 1
    [2, 1, 1, 1, 1, 0, 1, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 2, 0, 1]
]


class TestNaivebayes(unittest.TestCase):
    def test_separate_by_class(self):
        nb = Learner(22, 2)
        classes = nb.separate_by_class(default_dataset)
        self.assertEqual(default_dict, classes)

    def test_train(self):
        nb = Learner(22, 2)
        nb.train(default_dataset)
        self.assertEqual(default_feature_class, nb.feature_class)
        self.assertEqual(nb.classes[0], 2)
        self.assertEqual(nb.classes[1], 2)

    def test_classify(self):
        nb = Learner(22, 2)
        instance = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        nb.train(default_dataset)
        print(nb.classify(instance))

    def test_read_csv(self):
        nb = Learner(22, 2)
        dataset = nb.read_csv('spect-orig.train.csv')
        for row in dataset[:5]:
            print(row)