import numpy as np
import pandas as pd
import math
from DecisionTreeNode import DecisionTreeNode


class DecisionTreeClassifier:
    
    def __init__(self):
        self.root = None
    
    def get_root(self):
        return self.root

    def get_y_dic(self, Y):
        dic = {}
        for i in Y:
            if i not in dic:
                dic[i] = 0
            dic[i] += 1
        return dic
    
    def plogp(self, pi):
        if pi == 0:
            return 0
        elif pi == 1:
            return 0
        else:
            return (- pi * math.log2(pi))

    def get_entropy(self, Y):
        Y_map = self.get_y_dic(Y)
        total = len(Y)

        entropy = 0
        for i in Y_map:
            pi = (Y_map[i] / total)
            entropy += self.plogp(pi)
        return entropy

    def get_info_gain(self, X, Y, i):
        uniq_vals = self.get_y_dic(X[:, i])
        total = len(uniq_vals)

        df = pd.DataFrame(X)
        df[df.shape[1]] = Y

        entropy_of_class = 1
        for j in uniq_vals:
            # subtracting entropy of attributes from entropy of class to calculate the gain
            dfj = df[df[i] == j]
            y = dfj[dfj.shape[1]-1]
            entropy_of_class -= ((uniq_vals[j]/total) * self.get_entropy(y))
        
        # finally entropy_of_class becomes equal to gain
        return entropy_of_class

    def create_tree(self, X, Y, features, level, classes):
        '''
        when to Stop in a decision Tree:
        1. N(Features) = 0
        2. len(set(y)) = 1 or pure class is obtained
        '''
        if level > len(features):
            Y_map = self.get_y_dic(Y)
            max_freq = -1
            class_m = -1
            for i in Y_map:
                if Y_map[i] > max_freq:
                    max_freq = Y_map[i]
                    class_m = i
            output = class_m

            print(f"Level : {level}")
            for i in classes:
                if i in Y_map:
                    print(f"Count of {i} : {Y_map[i]}")
                else:
                    print(f"Count of {i} : 0")
            # print(f"information gain is :{self.get_entropy(Y)}")
            print("Maximum depth reached.")
            return DecisionTreeNode(None, output)

        if len(features) == 0:
            # here we need to assign the maximum ne the class
            Y_map = self.get_y_dic(Y)
            max_freq = -1
            class_m = -1
            for i in Y_map:
                if Y_map[i] > max_freq:
                    max_freq = Y_map[i]
                    class_m = i
            output = class_m

            print(f"Level : {level}")
            for i in classes:
                if i in Y_map:
                    print(f"Count of {i} : {Y_map[i]}")
                else:
                    print(f"Count of {i} : 0")
            # print(f"information gain is :{self.get_entropy(Y)}")
            print("Reached Leaf Node")
            return DecisionTreeNode(None, output)
        
        if len(set(Y)) == 1:
            Y_map = self.get_y_dic(Y)
            print()
            print(f"Level : {level}")
            for i in classes:
                if i in Y_map:
                    print(f"Count of {i} : {Y_map[i]}")
                else:
                    print(f"Count of {i} : 0")
            print("Current information gain is 0")
            print("Reached Leaf Node")
            return DecisionTreeNode(None, Y[0])
        
        # Stopping conditions are now managed
        # now handling for general condition

        max_gain = - int(1e7)
        sel_fe = None
        for i in features:
            i_gain = self.get_info_gain(X, Y, i)
            if i_gain > max_gain :
                max_gain = i_gain
                sel_fe = i
        
        # printing the splitting info for the node and the output class
        output = None
        max_val = -int(1e7)
        Y_map = self.get_y_dic(Y)
        print()
        print(f"Level is : {level}")
        for i in classes:
            if i in Y_map:
                print(f"Count of {i} : {Y_map[i]}")
                if Y_map[i] > max_val:
                    output = i
            else:
                print(f"Count of {i} : 0")
        # print(f"Current information : {self.get_entropy(Y)}")
        print(f"Splitting on the feature : {output} and with Gain : {max_gain}")

        # now performing the splitting by removing the feature of the features matrix
        indexofFeature = features.index(sel_fe)
        features.remove(sel_fe)
        X_unique_vals = set(X[:,sel_fe])
        df = pd.DataFrame(X)
        #  adding Y also to the data frame to perform the split
        df[df.shape[1]] = Y

        # creating the current node
        current_node = DecisionTreeNode(sel_fe, output)

        # splitting
        for i in X_unique_vals:
            # getting the data frame with x[sel_fe] = i
            dfi = df[df[sel_fe] == i]

            # the recursion provides childnodes
            node = (self.create_tree(dfi.iloc[:, 0:dfi.shape[1] - 1].values, dfi.iloc[:,dfi.shape[1]-1].values, features, level+1, classes))
            current_node.add_child(i, node)

        # again inserting the feature in its place after the splitting's affect has been captured so that now it can be used again
        features.insert(indexofFeature,sel_fe)

        # returning the node hence created
        return current_node

    def fit(self, X, Y):
        features = [i for i in range(X.shape[-1])]
        classes =  set(Y)
        self.root = self.create_tree(X, Y, features, 0,classes)

    def predict_for_xi(self, Xi, root):
        if len(root.children) == 0:
            return root.output
        split_fe = Xi[root.feature]
        if split_fe not in root.children:
            return root.output
        return self.predict_for_xi(Xi, root.children[split_fe])

    def predict(self, xtest):
        Y = np.zeros(len(xtest))
        for i in range(len(xtest)):
            Y[i] = self.predict_for_xi(xtest[i], self.root)
        return Y
    
    def score(self, ypred, ytest):
        count = 0
        for i in range(len(ypred)):
            if ypred[i] == ytest[i]:
                count += 1
        return count/len(ypred)
