import unittest
from knn import Learner

default_dataset = [
    # 23 numbers, 22 features, 1st is the class (1 of 2)
    [1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
]

small_set = [
    [1, 4, 1],
    [2, 8, 4]
]


class TestKnn(unittest.TestCase):
    def test_distance(self):
        kn = Learner()
        self.assertEqual(5, kn.distance(small_set[0], small_set[1]))

    def test_ham(self):
        kn = Learner()
        self.assertEqual(14, kn.distance_h(default_dataset[0], default_dataset[2]))

    def test_neighbors(self):
        kn = Learner()
        neighbors = kn.neighbors(default_dataset[0], default_dataset[1:])
        self.assertEqual([(13, 0), (14, 1), (11, 0)], neighbors)

    def test_k_nearest(self):
        kn = Learner()
        k_neighbors = kn.k_nearest(default_dataset[0], default_dataset[1:], 2)
        self.assertEqual([(11, 0), (13, 0)], k_neighbors)

    def test_classify(self):
        kn = Learner()
        self.assertEqual(0, kn.classify(default_dataset[0], default_dataset[1:], 2))


if __name__ == '__main__':
    unittest.main()
