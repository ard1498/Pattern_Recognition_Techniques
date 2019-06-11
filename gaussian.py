import numpy as np
from pprint import pprint
import math

class Gaussian:
    def __init__(self,sigma = 1, m = 1):
        self.result = {}
        self.sigma = sigma
        self.K = 1/((2*3.14)**(m/2) * self.sigma**m)
        self.classes = set()
    
    def train(self, X, Y):
        self.classes = set(Y)
        no_of_features = len(X[0])
        self.result['total'] = len(X)
        for i in self.classes:
            self.result[i] = {}
            for j in range(no_of_features):
                self.result[i][j] = list()
                
        for i in range(len(X)):
            self.result[Y[i]]['total'] = (Y == Y[i]).sum()
            for j in range(no_of_features):
                self.result[Y[i]][j].append(X[i][j])
        
        pprint(self.result)
        return
    
    def get_gaussian_probab(self, X_test):
        # calculating for every classes
        no_of_features = X_test.shape[-1]
        final_probabilities = {}
        for i in self.classes:
            final_post_probab = 1
            for j in range(no_of_features):
                final_expr = 0
                for k in self.result[i][j]:
                    final_expr += math.exp(-((k - X_test[j])**2)/(2*(self.sigma**2)))
                final_post_probab *= final_expr
            final_probabilities[i] = final_post_probab
        return final_probabilities
    
    def predict(self, X_test):
        # calculating priori of classes
        priori = {}
        for i in self.classes:
            priori[i] = self.result[i]['total'] / self.result['total']
        # get the posteieri probabilities
        final_post_probability = self.get_gaussian_probab(X_test)
        
        for i in self.classes:
            print('probability of class ' + i + ' is :' + str(final_post_probability[i] * priori[i]))
        
        return
        

def main():
    m = int(input("enter the number of independent attributes:"))
    n = int(input("enter the number of data points:"))

    features = []
    for j in range(m):
        features.append(str(input('enter f'+str(j) + ' name:')))

    outputs = []
    X = []
    for i in range(n):
        Xi= []
        print('enter the '+str(i)+' data point:')
        for j in range(m):
            Xi.append(float(input('Value of ' + features[j] + ':')))
        X.append(Xi)
        outputs.append(str(input('enter the class for the data point:')))

    print('____________________________________________________________________________')
    X = np.array(X)
    Y = np.array(outputs)
    for i in range(n):
        for j in range(m):
            print(X[i][j], end=' ')
        print(outputs[i])
    clf = Gaussian()
    print(clf.train(X, Y))
    
    # clf.printdictionaries()
    print('_____________________________________________________________________________')
    print('enter the test data point:')
    X_test = []
    for j in range(len(features)):
        X_test.append(float(input('Value of ' + features[j] + ':')))
    X_test = np.array(X_test)    
    print("x test is :")
    print(X_test)
    print("__________________Now the prediction and results _________________________")
    clf.predict(X_test)
    # print('class is :' + clf.predict(x, confidence))
    
if __name__ == '__main__':
    main()