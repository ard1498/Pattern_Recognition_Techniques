from NaiveBayes import NaiveBayes
import numpy as np
from collections import Counter

class Clean:
    def __init__(self):
        self.features = {}
        self.lam_error = []

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
    
    def getErrorFunction(self, no_of_classes, no_of_actions, actions, classes):
        self.lam_error = [[0 for j in range(no_of_classes)] for i in range(no_of_actions)]
        for i in range(no_of_actions):
            for j in range(no_of_classes):
                self.lam_error[i][j] = int(input(f'enter lambda for ({actions[i]}, {classes[j]}):'))
                
    def getRisk(self, class_probabs, actions):
        no_of_actions = len(self.lam_error)
        Risks = [0 for i in range(no_of_actions)]
        for i in range(len(Risks)):
            for j in range(len(class_probabs)):
                Risks[i] += class_probabs[j]*self.lam_error[i][j]
        min_val,action = 10**6,0
        for i in range(len(Risks)):
            if Risks[i] < min_val:
                min_val = Risks[i]
                action = i
        print(Risks)
        print('the predicted action is :' + actions[action])

def main():
    n = int(input('enter no of training data sentences :'))
    X,Y = [],[]
    for i in range(n):
        words_i = input('enter ' + str(i+1)+'th sentence:').strip().split(' ')
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
    no_of_classes = len(set(Y))
    no_of_actions = int(input('enter the no of actions :'))
    actions = []
    for i in range(no_of_actions):
        action = input('enter the name of action :')
        actions.append(action)
    clean_obj.getErrorFunction(no_of_classes, no_of_actions, actions, list(set(Y)))
    for i in range(m):
        best_class, class_probabilities = clf.predict_text_for_Risk(Xtest_trans[i])
        print('the class it belong :' + best_class)
        clean_obj.getRisk(class_probabilities,actions)


if __name__ == '__main__':
    main()
