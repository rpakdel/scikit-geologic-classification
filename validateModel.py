import numpy as np
from sklearn.metrics import confusion_matrix
from classification_utilities import display_cm, display_adj_cm

#%% [markdown]
# Now that the model has been trained on our data, we can use it to predict the facies of the feature vectors in the 
# test set.  Because we know the true facies labels of the vectors in the test set, we can use the results to evaluate 
# the accuracy of the classifier.
#%%

def accuracy(conf):
    total_correct = 0.
    nb_classes = conf.shape[0]
    for i in np.arange(0,nb_classes):
        total_correct += conf[i][i]
    acc = total_correct/sum(sum(conf))
    return acc

def accuracy_adjacent(conf, adjacent_facies):
    nb_classes = conf.shape[0]
    total_correct = 0.
    for i in np.arange(0,nb_classes):
        total_correct += conf[i][i]
        for j in adjacent_facies[i]:
            total_correct += conf[i][j]
    return total_correct / sum(sum(conf))


def validate(clf, x_test, y_test, facies_labels, adjacent_facies):
    predicted_labels = clf.predict(x_test)

    # We need some metrics to evaluate how good our classifier is doing.  
    # A [confusion matrix](http://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/) is a table that can be 
    # used to describe the performance of a classification model.  
    # [Scikit-learn](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html) allows us 
    # to easily create a confusion matrix by supplying the actual and predicted facies labels.
    # 
    # The confusion matrix is simply a 2D array.  The entries of confusion matrix `C[i][j]` are equal to the number of 
    # observations predicted to have facies `j`, but are known to have facies `i`.  
    # 
    # To simplify reading the confusion matrix, a function has been written to display the matrix along with facies labels 
    # and various error metrics.  See the file `classification_utilities.py` in this repo for the `display_cm()` function.
    print('CONFUSION MATRIX')
    conf = confusion_matrix(y_test, predicted_labels)
    #display_cm(conf, facies_labels, display_metrics=True, hide_zeros=True)

    #%% [markdown]
    # The rows of the confusion matrix correspond to the actual facies labels.  The columns correspond to the labels 
    # assigned by the classifier.  For example, consider the first row. For the feature vectors in the test set that 
    # actually have label `SS`, 23 were correctly indentified as `SS`, 21 were classified as `CSiS` and 2 were 
    # classified as `FSiS`.
    # 
    # The entries along the diagonal are the facies that have been correctly classified.  Below we define two functions 
    # that will give an overall value for how the algorithm is performing.  The accuracy is defined as the number of 
    # correct classifications divided by the total number of classifications.
    #%%

    #%% [markdown]
    # As noted above, the boundaries between the facies classes are not all sharp, and some of them blend into one another.  
    # The error within these 'adjacent facies' can also be calculated.  
    print()
    print('Seam classification accuracy = %f' % accuracy(conf))
    #print('Adjacent facies classification accuracy = %f' % accuracy_adjacent(conf, adjacent_facies))

    #precisionRecall(conf, facies_labels, adjacent_facies)

#%% [markdown]
# [Precision and recall](https://en.wikipedia.org/wiki/Precision_and_recall) are metrics that give more insight into 
# how the classifier performs for individual facies.  Precision is the probability that given a classification result 
# for a sample, the sample actually belongs to that class.  Recall is the probability that a sample will be correctly 
# classified for a given class.
# 
# Precision and recall can be computed easily using the confusion matrix.  The code to do so has been added to the 
# `display_confusion_matrix()` function:
#%%

def precisionRecall(conf, facies_labels, adjacent_facies):
    #display_cm(conf, facies_labels, display_metrics=True, hide_zeros=True)

    #%% [markdown]
    # To interpret these results, consider facies `SS`.  In our test set, if a sample was labeled `SS` the probability the 
    # sample was correct is 0.8 (precision).  If we know a sample has facies `SS`, then the probability it will be 
    # correctly labeled by the classifier is 0.78 (recall).  It is desirable to have high values for both precision and 
    # recall, but often when an algorithm is tuned to increase one, the other decreases.  The 
    # [F1 score](https://en.wikipedia.org/wiki/Precision_and_recall#F-measure) combines both to give a single measure of 
    # relevancy of the classifier results.
    # 
    # These results can help guide intuition for how to improve the classifier results.  For example, for a sample with 
    # facies `MS` or mudstone, it is only classified correctly 57% of the time (recall).  Perhaps this could be improved 
    # by introducing more training samples.  Sample quality could also play a role.  Facies `BS` or bafflestone has the 
    # best `F1` score and relatively few training examples.  But this data was handpicked from other wells to provide 
    # training examples to identify this facies.
    # 
    # We can also consider the classification metrics when we consider misclassifying an adjacent facies as correct: 
    #%%
    print('ADJACENT FACIES CONFUSION MATRIX')
    display_adj_cm(conf, facies_labels, adjacent_facies, display_metrics=True, hide_zeros=True)

    #%% [markdown]
    # Considering adjacent facies, the `F1` scores for all facies types are above 0.9, except when classifying `SiSh` 
    # or marine siltstone and shale.  The classifier often misclassifies this facies (recall of 0.66), most often as 
    # wackestone. 
    # 
    # These results are comparable to those reported in Dubois et al. (2007).