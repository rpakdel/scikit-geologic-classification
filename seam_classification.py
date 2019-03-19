import loadData as data
from prepareData import scaleData
from modelSelection import splitData
from validateModel import validate
from blindData import blindTest
import matplotlib.pyplot as plt
from pandas.plotting import table
from exploreData import exploreData

from trainData import trainUsingSVMC
from trainData import trainUsingSVMCWithParameters
from trainData import trainUsingMPLC
from trainData import traingUsingKNeighborsC
from trainData import traingUsingRandomForestC

# load and prepare the data
# removes empty entries and maps labels and colours
# also creates some blind data for validation
training_data, blind = data.loadAndPrepareData()

# explore the data
#exploreData(training_data, data.seam_labels, data.seam_colors)

# Now we extract just the feature variables we need to perform the classification.  The predictor variables are the 
# five wireline values and two geologic constraining variables. We also get a vector of the facies labels that 
# correspond to each feature vector.
training_data = data.removeEmptyValues(training_data)

# the known labels
correct_facies_labels = training_data['SEAM'].values
# features used to determine the labels
feature_vectors = training_data.drop(['DHID', 'FROM', 'TO','ASEAM','SEAM'], axis=1)
# descriptive statistics of the features
featureStats = feature_vectors.describe()
print(featureStats)

# standardize the data
# keep the scaler for later for test and blind data
scaler, scaled_features = scaleData(feature_vectors)

# split the features between training and testing data
x_train, x_test, y_train, y_test = splitData(scaled_features, correct_facies_labels)

clf = trainUsingSVMC(x_train, y_train)
print()
print('######### SUPPORT VECTOR MACHINE #########')
validate(clf, x_test, y_test, data.seam_labels, None)

# The best accuracy on the cross validation error curve was achieved for `gamma = 1`, and `C = 10`. 
#  We can now create and train an optimized classifier based on these parameters:
#clf = trainUsingSVMWithParameters(x_train, y_train, C=10, gamma=1)
#print('######## Training data, optimized parameters ##########')
#validate(clf, x_test, y_test, data.facies_labels, data.adjacent_facies)

#clf = trainUsingMPLC(x_train, y_train)
#print()
#print('######### MULTIPLAYER PERCEPTRON #########')
#validate(clf, x_test, y_test, data.facies_labels, data.adjacent_facies)

#clf = traingUsingKNeighborsC(x_train, y_train)
#print()
#print('######### K NEAREST NEIGHBOURS CLASSIFIER #########')
#validate(clf, x_test, y_test, data.facies_labels, data.adjacent_facies)

#clf = traingUsingRandomForestC(x_train, y_train, n_estimators=12)
#print()
#print('######### RANDOM FOREST CLASSIFIER #########')
#validate(clf, x_test, y_test, data.seam_labels, None)


# now predict a new well
# TODO: fix the compare facies plot
blind = data.removeEmptyValues(blind)
blindTest(blind, scaler, clf, data.seam_labels, data.seam_colors, None)
plt.show()
