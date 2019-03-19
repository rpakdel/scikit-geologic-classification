from sklearn.model_selection import train_test_split
import numpy as np

#%% [markdown]
# Scikit also includes a handy function to randomly split the training data into training and test sets.  The test 
# set contains a small subset of feature vectors that are not used to train the network.  Because we know the true 
# facies labels for these examples, we can compare the results of the classifier to the actual facies and determine 
# the accuracy of the model.  Let's use 20% of the data for the test set.

def splitData(scaled_features, correct_facies_labels):
    X_train, X_test, y_train, y_test = train_test_split(scaled_features, correct_facies_labels, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


#%% [markdown]
# ## Model parameter selection
# 
# The classifier so far has been built with the default parameters.  However, we may be able to get improved 
# classification results with optimal parameter choices.
# 
# We will consider two parameters.  The parameter `C` is a regularization factor, and tells the classifier how much we 
# want to avoid misclassifying training examples.  A large value of C will try to correctly classify more examples 
# from the training set, but if `C` is too large  it may 'overfit' the data and fail to generalize when classifying new data. If `C` is too small then the model will not be good at fitting outliers and will have a large error on the training set.
# 
# The SVM learning algorithm uses a kernel function to compute the distance between feature vectors.  Many kernel 
# functions exist, but in this case we are using the radial basis function `rbf` kernel (the default).  The `gamma` 
# parameter describes the size of the radial basis functions, which is how far away two vectors in the feature space 
# need to be to be considered close.
# 
# We will train a series of classifiers with different values for `C` and `gamma`.  Two nested loops are used to train 
# a classifier for every possible combination of values in the ranges specified.  The classification accuracy is 
# recorded for each combination of parameter values.  The results are shown in a series of plots, so the parameter 
# values that give the best classification accuracy on the test set can be selected.
# 
# This process is also known as 'cross validation'.  Often a separate 'cross validation' dataset will be created in a
# ddition to the training and test sets to do model selection.  For this tutorial we will just use the test set to 
# choose model parameters.
#%%

def doModelSelection():
    C_range = np.array([.01, 1, 5, 10, 20, 50, 100, 1000, 5000, 10000])
    gamma_range = np.array([0.0001, 0.001, 0.01, 0.1, 1, 10])
    
    fig, axes = plt.subplots(3, 2, 
                        sharex='col', sharey='row',figsize=(10,10))
    plot_number = 0
    for outer_ind, gamma_value in enumerate(gamma_range):
        row = int(plot_number / 2)
        column = int(plot_number % 2)
        cv_errors = np.zeros(C_range.shape)
        train_errors = np.zeros(C_range.shape)
        for index, c_value in enumerate(C_range):
            
            clf = svm.SVC(C=c_value, gamma=gamma_value)
            clf.fit(X_train,y_train)
            
            train_conf = confusion_matrix(y_train, clf.predict(X_train))
            cv_conf = confusion_matrix(y_test, clf.predict(X_test))
        
            cv_errors[index] = accuracy(cv_conf)
            train_errors[index] = accuracy(train_conf)

        ax = axes[row, column]
        ax.set_title('Gamma = %g'%gamma_value)
        ax.semilogx(C_range, cv_errors, label='CV error')
        ax.semilogx(C_range, train_errors, label='Train error')
        plot_number += 1
        ax.set_ylim([0.2,1])
        
    ax.legend(bbox_to_anchor=(1.05, 0), loc='lower left', borderaxespad=0.)
    fig.text(0.5, 0.03, 'C value', ha='center',
             fontsize=14)
             
    fig.text(0.04, 0.5, 'Classification Accuracy', va='center', 
             rotation='vertical', fontsize=14)
