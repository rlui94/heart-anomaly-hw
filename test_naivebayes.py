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
        feature_class, classes = nb.train(default_dataset)
        self.assertEqual(default_feature_class, feature_class)
        self.assertEqual(classes[0], 2)
        self.assertEqual(classes[1], 2)

