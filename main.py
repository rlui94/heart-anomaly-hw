from naivebayes import Learner

if __name__ == '__main__':
    nb = Learner()
    training_data = nb.read_csv_train('spect-orig.train.csv')
    testing_data = nb.read_csv_test('spect-orig.test.csv')
    nb.train(training_data)
    nb.classify_check(testing_data)
