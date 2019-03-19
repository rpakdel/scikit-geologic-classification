from sklearn import preprocessing

#%% [markdown]
# Scikit includes a [preprocessing](http://scikit-learn.org/stable/modules/preprocessing.html) module that can 
# 'standardize' the data (giving each variable zero mean and unit variance, also called *whitening*).  
# Many machine learning algorithms assume features will be standard normally distributed data 
# (ie: Gaussian with zero mean and unit variance).  The factors used to standardize the training set must be applied 
# to any subsequent feature set that will be input to the classifier.  The `StandardScalar` class can be fit to the 
# training set, and later used to standardize any training data.
#%%

def scaleData(feature_vectors):
    scaler = preprocessing.StandardScaler().fit(feature_vectors)
    scaled_features = scaler.transform(feature_vectors)
    return (scaler, scaled_features)