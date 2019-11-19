from naivebayes import Learner
# from knn import Learner
FILEFLAG = 1

def file_switch(flag):
    switcher = {
        1: ('spect-orig.train.csv', 'spect-orig.test.csv', 'orig'),
        2: ('spect-itg.train.csv', 'spect-itg.test.csv', 'itg'),
        3: ('spect-resplit.train.csv', 'spect-resplit.test.csv', 'resplit'),
    }
    return switcher.get(flag, "Invalid flag")


if __name__ == '__main__':
    machine = Learner()
    print(machine.solve())

