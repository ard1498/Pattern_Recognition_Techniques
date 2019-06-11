import numpy as np

class MEstimate:
    def __init__(self):
        self.result = {}
        self.prior_expectation = 1/2

    def compute_probability(self, x, curclass, m = 0):
        priori_curclass = self.result[curclass]['totalcount'] / self.result['totalcount']

        #now for conditional probability
        num_features = len(self.result[curclass].keys()) - 1
        picond_probab = 1
        for j in range(num_features):
            xj = x[j]
            count_cur_class_with_val_xj = self.result[curclass][j][xj] + m*self.prior_expectation
            count_cur_class = self.result[curclass]['totalcount'] + m
            cond_probab = count_cur_class_with_val_xj/count_cur_class
            picond_probab *= cond_probab

        return priori_curclass*picond_probab
    
    def train(self, X, Y):
        class_values = set(Y)
        self.result['totalcount'] = len(Y)
        for currentclass in class_values:
            self.result[currentclass] = {}
            num_features = X.shape[1]
            curr_rows_required = (Y == currentclass)
            X_cur = X[curr_rows_required]
            Y_cur = Y[curr_rows_required]
            self.result[currentclass]['totalcount'] = len(Y_cur)
            for j in range(num_features):
                self.result[currentclass][j] = {}
                all_possible_values = set(['0','1'])
                for curr_val_X in all_possible_values:
                    self.result[currentclass][j][curr_val_X] = (curr_val_X == X_cur[:,j]).sum()
        return 'Training Successfull'

    def predict(self, x, m = 0):
        classes = self.result.keys()
        best_p = -1
        best_class = -1
        for curclass in classes:
            if curclass != 'totalcount':
                pcurclass = self.compute_probability(x, curclass, m)
                if pcurclass > best_p:
                    best_p = pcurclass
                    best_class = curclass
        return best_class
    
    def printdictionaries(self):
        print(self.result)
        
def main():
    m = int(input("enter the number of features:"))
    n = int(input("enter the number of data points:"))

    features = []
    for j in range(m):
        features.append(str(input('enter f'+str(j) + ' name:')))

    confidence = int(input('enter confidence value : '))
    outputs = []
    X = []
    for i in range(n):
        Xi= []
        print('enter the '+str(i)+' data point:')
        for j in range(m):
            Xi.append(str(input('Object ' + features[j] + '? 1-yes/0-no:')))
        X.append(Xi)
        print('enter the class for the data point:')
        outputs.append(str(input()))

    print('________________________________________________________________________')
    X = np.array(X)
    Y = np.array(outputs)
    for i in range(n):
        for j in range(m):
            print(X[i][j], end=' ')
        print(outputs[i])
    clf = MEstimate()
    print(clf.train(X, Y))
    clf.printdictionaries()

    print('enter the test data point:')
    x = []
    for j in range(len(features)):
        x.append(input('Object ' + features[j] + ' ? 1-Yes,0-No :'))
        
    print("__________________Now the prediction and results _________________________")
    print('class is :' + clf.predict(x, confidence))

if __name__ == '__main__':
    main()


'''
A:\Anaconda3\python.exe C:/Users/ANIRUDH/Desktop/SEM6/PatternRLab/mestimate.py
enter the number of features:4
enter the number of data points:5
enter f0 name:red
enter f1 name:blue
enter f2 name:green
enter f3 name:yellow
enter confidence value : 6
enter the 0 data point:
Object red? 1-yes/0-no:1
Object blue? 1-yes/0-no:1
Object green? 1-yes/0-no:0
Object yellow? 1-yes/0-no:0
enter the class for the data point:
1
enter the 1 data point:
Object red? 1-yes/0-no:1
Object blue? 1-yes/0-no:0
Object green? 1-yes/0-no:1
Object yellow? 1-yes/0-no:0
enter the class for the data point:
1
enter the 2 data point:
Object red? 1-yes/0-no:0
Object blue? 1-yes/0-no:0
Object green? 1-yes/0-no:0
Object yellow? 1-yes/0-no:1
enter the class for the data point:
2
enter the 3 data point:
Object red? 1-yes/0-no:0
Object blue? 1-yes/0-no:0
Object green? 1-yes/0-no:1
Object yellow? 1-yes/0-no:1
enter the class for the data point:
2
enter the 4 data point:
Object red? 1-yes/0-no:0
Object blue? 1-yes/0-no:1
Object green? 1-yes/0-no:0
Object yellow? 1-yes/0-no:0
enter the class for the data point:
1
________________________________________________________________________
1 1 0 0 1
1 0 1 0 1
0 0 0 1 2
0 0 1 1 2
0 1 0 0 1
Training Successfull
{'totalcount': 5, '2': {'totalcount': 2, 0: {'0': 2, '1': 0}, 1: {'0': 2, '1': 0}, 2: {'0': 1, '1': 1}, 3: {'0': 0, '1': 2}}, '1': {'totalcount': 3, 0: {'0': 1, '1': 2}, 1: {'0': 1, '1': 2}, 2: {'0': 2, '1': 1}, 3: {'0': 3, '1': 0}}}
enter the test data point:
Object red ? 1-Yes,0-No :1
Object blue ? 1-Yes,0-No :1
Object green ? 1-Yes,0-No :1
Object yellow ? 1-Yes,0-No :1
__________________Now the prediction and results _________________________
class is :1

Process finished with exit code 0

'''