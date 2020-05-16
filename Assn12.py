from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn import tree, ensemble
from sklearn import model_selection 
from sklearn.ensemble import BaggingClassifier 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from mpl_toolkits import mplot3d
import graphviz
#matplotlib inline
import numpy as np
import matplotlib.pyplot as plt


class Evaluation:
    def __init__(self):
        self.cancer = load_breast_cancer()
        self.X_train, self.X_test, self.y_train, self.y_test \
            = train_test_split(self.cancer.data, self.cancer.target, test_size=0.3)
        self.tree_score = 0
        self.bagging_score = []
        self.boost_score = []
        self.forest_score = {}

    def decision_tree(self):
        clf = tree.DecisionTreeClassifier(criterion="gini")
        clf = clf.fit(self.X_train, self.y_train)
        self.tree_score = (clf.score(self.X_test, self.y_test))
        dot_data = tree.export_graphviz(clf, out_file=None, feature_names=self.cancer.feature_names)
        graph = graphviz.Source(dot_data)
        graph.render("Cancer")
    def bagging(self):
        seed = 8
        kfold = model_selection.KFold(n_splits = 30, random_state = seed) 
        base_cls = DecisionTreeClassifier(criterion="gini") 
        num_trees = 500
        model = BaggingClassifier(base_estimator = base_cls, n_estimators = num_trees, random_state = seed)
        self.bagging_score.append(model_selection.cross_val_score(model, self.X_train, self.y_train, cv = kfold))
        print("Bagging Scores")
        plt.plot(30, 1, self.bagging_score[0])
        plt.show()
        
    def forest(self):
        for x in range(1,12):
            for y in range(1,12):
                clf = ensemble.RandomForestClassifier(n_estimators=x, max_features=y)
                clf.fit(self.X_train, self.y_train)
                self.forest_score.update({(x,y): clf.score(self.X_test, self.y_test)})
        print(self.forest_score)
        
        lists = sorted(self.forest_score.items()) # sorted by key, return a list of tuples
        x, z = zip(*lists) # unpack a list of pairs into two tuples
        x, y = zip(*x)
        fig = plt.figure()
        
        ax = plt.axes(projection='3d')
        zdata = z
        xdata = x
        ydata = y
        ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens');
                
        
    def boost(self):
        seed = 8
        kfold = model_selection.KFold(n_splits = 30, random_state = seed) 
        base_cls = DecisionTreeClassifier(criterion="gini") 
        num_trees = 500
        model = AdaBoostClassifier(base_estimator = base_cls, n_estimators = num_trees, random_state = seed)
        self.boost_score.append(model_selection.cross_val_score(model, self.X_train, self.y_train, cv = kfold))
        #print("accuracy :", self.bagging_score) 
        print("Boost Scores")
        plt.plot(30, 1, self.boost_score[0])
        plt.show()
#
#
#    def summary(self):


if __name__ == '__main__':
   exp = Evaluation()
#   exp.decision_tree()
#   print("exp.tree_score", exp.tree_score)
#   exp.bagging()
   exp.forest()
#   exp.boost()
#   exp.summary()