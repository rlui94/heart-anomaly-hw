from naivebayes import Learner
FILEFLAG = 1

def file_switch(flag):
    switcher = {
        1: ('spect-orig.train.csv', 'spect-orig.test.csv', 'orig'),
        2: ('spect-itg.train.csv', 'spect-itg.test.csv', 'itg'),
        3: ('spect-resplit.train.csv', 'spect-resplit.test.csv', 'resplit'),
    }
    return switcher.get(flag, "Invalid flag")


if __name__ == '__main__':
    nb = Learner()
    set = file_switch(FILEFLAG)
    training_data = nb.read_csv_train(set[0])
    testing_data = nb.read_csv_test(set[1])
    nb.train(training_data)
    str = nb.classify_check(testing_data)
    print('%s %s' % (set[2], str))

