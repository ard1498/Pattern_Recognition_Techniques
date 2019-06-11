from NaiveBayes import NaiveBayes
import numpy as np
from collections import Counter

class Clean:
    def __init__(self):
        self.features = {}

    def print_features(self):
        if len(self.features) > 0 :
            print(self.features)
        else:
            print('features are not extracted yet.')

    def feature_extract(self, X, n):
        features = []
        for i in range(n):
            features += list(X[i])
        self.features = set(features)
        return

    def transform_X(self, X, n):
        for i in range(n):
            c = Counter(X[i])
            new_X_i = []
            for j in self.features:
                if j not in c:
                    new_X_i.append(0)
                else:
                    new_X_i.append(c[j])
            X[i] = new_X_i
        return np.array(X)
    
    def transform_Y(self, Y):
        return np.array(Y)

def main():
    n = int(input('enter no of training data sentences :'))
    X,Y = [],[]
    for i in range(n):
        words_i = input('enter ' + str(i)+'th sentence:').strip().split(' ')
        X.append(words_i)
        Y.append(input('enter the class of sentence :').strip())
    
    clean_obj = Clean()
    clean_obj.feature_extract(X, n)
    clean_obj.print_features()
    X = clean_obj.transform_X(X, n)
    Y = clean_obj.transform_Y(Y)
    print('X is :',X)
    print('Y is :',Y)

    clf = NaiveBayes()
    clf.train_text(X, Y)
    clf.printdictionaries()
    
    m = int(input('enter number of testing entries:'))
    Xtest = []
    for i in range(m):
        test_words_i = input('enter ' + str(i+1)+'th sentence:').strip().split(' ')
        Xtest.append(test_words_i)
    Xtest_trans = clean_obj.transform_X(Xtest, m)
    print(Xtest_trans)
    for i in range(m):
        print(clf.predict_text(Xtest_trans[i]))


if __name__ == '__main__':
    main()
