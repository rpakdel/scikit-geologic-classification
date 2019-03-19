from sklearn import svm
from sklearn.neural_network import MLPClassifier 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

#%% [markdown]
# ## Training the SVM classifier
# 
# Now we use the cleaned and conditioned training set to create a facies classifier.  
# As mentioned above, we will use a type of machine learning model known as a 
# [support vector machine](https://en.wikipedia.org/wiki/Support_vector_machine).  The SVM is a map of the feature 
# vectors as points in a multi dimensional space, mapped so that examples from different facies are divided by a clear 
# gap that is as wide as possible.  
# 
# The SVM implementation in 
# [scikit-learn](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC) takes a 
# number of important parameters.  First we create a classifier using the default settings.  
#%%

def trainUsingSVMC(x_train, y_train):
    clf = svm.SVC(gamma='scale')
    clf.fit(x_train,y_train)
    return clf

def trainUsingSVMCWithParameters(x_train, y_train, C, gamma):
    clf = svm.SVC(C=10, gamma=1)        
    clf.fit(x_train, y_train)
    return clf

def trainUsingMPLC(x_train, y_train):
    clf = MLPClassifier()
    clf.fit(x_train,y_train)
    return clf

def traingUsingKNeighborsC(x_train, y_train):
    clf = KNeighborsClassifier()
    clf.fit(x_train, y_train)
    return clf

def traingUsingRandomForestC(x_train, y_train, n_estimators):
    clf = RandomForestClassifier(n_estimators=n_estimators)
    clf.fit(x_train, y_train)
    return clf
