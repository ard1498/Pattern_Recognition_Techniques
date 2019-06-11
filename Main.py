import sklearn.datasets as dt
import numpy as np
from sklearn.model_selection import train_test_split
from DecisionTree import DecisionTreeClassifier
from Print_Decision_Tree import get_tree_pdf


def get_labelled_data(column_data):
    second_limit = column_data.mean()
    first_limit = 0.5 * second_limit
    third_limit = 1.5 * second_limit
    for i in range(len(column_data)):
        cd = column_data[i]
        if cd <= first_limit:
            column_data[i] = 0
        elif cd <= second_limit:
            column_data[i] = 1
        elif cd <= third_limit:
            column_data[i] = 2
        else:
            column_data[i] = 3
    return column_data


def main():
    # loading the wine_dataset
    wine_ds = dt.load_wine()
    # iris_ds = dt.load_iris()
    X = wine_ds.data
    Y = wine_ds.target
    # X = iris_ds.data
    # Y = iris_ds.target
    # X.head(5),Y.head(5)
    
    # X = np.array([[0,1,0],
    #               [0,0,0],
    #               [0,0,1],
    #               [1,1,0],
    #               [1,1,1],
    #               [1,0,1],
    #               [1,0,0],
    #               [2,1,0],
    #               [2,0,1],
    #               [2,0,0]])
    # Y = np.array([0,
    #               0,
    #               0,
    #               0,
    #               0,
    #               1,
    #               1,
    #               1,
    #               1,
    #               1])
    
    # Now changing every feature column of X from linear
    # to class based
    for i in range(X.shape[1]):
        X[:, i] = np.array(get_labelled_data(X[:, i]))
    # Now we divide it in training and testing data
    x_train, x_test, y_train, y_test = train_test_split(X, Y, shuffle=True, test_size=0.30, random_state=3)

    # x_train, x_test, y_train, y_test = X, X, Y, Y
    
    # Now feeding the Training data to Decision tree classifier
    clf1 = DecisionTreeClassifier()
    clf1.fit(x_train, y_train)
    ypred = clf1.predict(x_test)
    print(f"Score is : {clf1.score(ypred, y_test)}")
    get_tree_pdf("wine_ds_n.pdf", clf1.get_root())


if __name__ == '__main__':
    main()
