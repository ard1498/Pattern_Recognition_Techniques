import numpy as np

class NaiveBayes:
    def __init__(self):
        self.result = {}

    def compute_probability(self, x, curclass):
        priori_curclass = self.result[curclass]['totalcount'] / self.result['totalcount']

        #now for conditional probability
        num_features = len(self.result[curclass].keys()) - 1
        picond_probab = 1
        for j in range(num_features):
            xj = x[j]
            count_cur_class_with_val_xj = self.result[curclass][j][xj] + 1
            count_cur_class = self.result[curclass]['totalcount'] + len(self.result[curclass][j].keys()) - 1
            cond_probab = count_cur_class_with_val_xj/count_cur_class
            picond_probab *= cond_probab

        return priori_curclass*picond_probab

    def compute_probability_text(self, x, curclass):
        priori_curclass = self.result[curclass]['totalcount'] / self.result['totalcount']
        #now for conditional probability
        num_features = len(self.result[curclass].keys()) - 2
        picond_probab = 1
        for j in range(num_features):
            xj = x[j]
            if xj != 0:
                count_cur_class_with_val_xj = self.result[curclass][j] + 1
                count_cur_class = self.result[curclass]['totalcount'] + self.result[curclass]['totalwords']
                cond_probab = xj*(count_cur_class_with_val_xj/count_cur_class)
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
                all_possible_values = set(X[:,j])
                for curr_val_X in all_possible_values:
                    self.result[currentclass][j][curr_val_X] = (curr_val_X == X_cur[:,j]).sum()
        return 'Training Successfull'

    def train_text(self, X, Y):
        class_values = set(Y)
        self.result['totalcount'] = len(Y)
        for currentclass in class_values:
            self.result[currentclass] = {}
            num_features = X.shape[1]
            curr_rows_required = (Y == currentclass)
            X_cur = X[curr_rows_required]
            Y_cur = Y[curr_rows_required]
            self.result[currentclass]['totalcount'] = len(Y_cur)
            self.result[currentclass]['totalwords'] = np.sum(X_cur)
            for j in range(num_features):
                self.result[currentclass][j] = np.sum(X_cur[:,j])
        return 'Training Successfull'

    def predict(self, x):
        classes = self.result.keys()
        best_p = -1
        best_class = -1
        for curclass in classes:
            if curclass != 'totalcount':
                pcurclass = self.compute_probability(x, curclass)
                if pcurclass > best_p:
                    best_p = pcurclass
                    best_class = curclass
        return best_class
    
    def predict_text(self, x):
        classes = self.result.keys()
        best_p = -1
        best_class = -1
        for curclass in classes:
            if curclass != 'totalcount' and curclass != 'totalwords':
                pcurclass = self.compute_probability_text(x, curclass)
                print('probability for class ' + curclass + ':' + str(pcurclass))
                if pcurclass > best_p:
                    best_p = pcurclass
                    best_class = curclass
        return best_class

    def predict_text_for_Risk(self, x):
        probabilities = []
        classes = self.result.keys()
        best_p = -1
        best_class = -1
        for curclass in classes:
            if curclass != 'totalcount' and curclass != 'totalwords':
                pcurclass = self.compute_probability_text(x, curclass)
                probabilities.append(pcurclass)
                print('probability for class ' + curclass + ':' + str(pcurclass))
                if pcurclass > best_p:
                    best_p = pcurclass
                    best_class = curclass
        return best_class,probabilities

    def printdictionaries(self):
        print(self.result)
        
def main():
    m = int(input("enter the number of features:"))
    n = int(input("enter the number of data points:"))

    features = []
    for j in range(m):
        features.append(str(input(f"enter f{j}:")))

    outputs = []
    X = []
    for i in range(n):
        Xi= []
        print(f'enter the {i} data point:')
        for j in range(m):
            Xi.append(str(input()))
        X.append(Xi)
        print('enter the class for the data point:')
        outputs.append(str(input()))

    X = np.array(X)
    Y = np.array(outputs)
    for i in range(n):
        for j in range(m):
            print(X[i][j], end=' ')
        print(outputs[i])
    clf = NaiveBayes()
    print(clf.train(X, Y))
    clf.printdictionaries()

    print('enter the test data point:')
    x = []
    for j in range(len(features)):
        x.append(input('enter ' + str(j) + ' feature :'))
    print(clf.predict(x))

if __name__ == '__main__':
    main()
